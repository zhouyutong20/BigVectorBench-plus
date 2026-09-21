#!/usr/bin/env python3
"""Controlled U1 HNSWlib predicate-cost experiment.

The two predicate paths intentionally share hnswlib's native Python callback
interface.  The precomputed path only replaces the callback body with a read
from the label-derived acceptance mask; the online path evaluates the original
32-round SplitMix64 predicate.  Query timing and native search counters are
collected in separate passes because the counters and callback timing are
instrumentation overhead, not part of the reported uninstrumented QPS.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np


RUN_ROOT = Path(os.environ.get("U1_PREDICATE_RUN_ROOT", Path(__file__).resolve().parent))
SRC_ROOT = Path(os.environ.get("U1_HNSWLIB_SRC", RUN_ROOT / "hnswlib_src"))
DATASET = Path(os.environ.get(
    "U1_DATASET",
    "<remote_artifact>",
))
INDEX_PATH = Path(os.environ.get(
    "U1_INDEX_PATH",
    RUN_ROOT / "index" / "hnswlib_M64_efc400.bin",
))
RAW_PATH = RUN_ROOT / "raw_passes.jsonl"
PROGRESS_PATH = RUN_ROOT / "progress.log"
SUMMARY_PATH = RUN_ROOT / "summary.json"

M = 64
EF_CONSTRUCTION = 400
K = 100
EFS = (200, 400, 800)
REPEATS = 5
WARMUP_ROUNDS = 2
WARMUP_QUERIES = 1000
QUERY_WORKERS = int(os.environ.get("U1_QUERY_WORKERS", "28"))
BUILD_THREADS = int(os.environ.get("U1_BUILD_THREADS", "56"))

MASK64 = (1 << 64) - 1
SM64_ADD = 0x9E3779B97F4A7C15
SM64_MUL1 = 0xBF58476D1CE4E5B9
SM64_MUL2 = 0x94D049BB133111EB


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def splitmix64_scalar(value: int, rounds: int) -> int:
    x = int(value) & MASK64
    for _ in range(int(rounds)):
        x = (x + SM64_ADD) & MASK64
        x = ((x ^ (x >> 30)) * SM64_MUL1) & MASK64
        x = ((x ^ (x >> 27)) * SM64_MUL2) & MASK64
        x = (x ^ (x >> 31)) & MASK64
    return x


def splitmix64_vector(values: np.ndarray, rounds: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        for _ in range(int(rounds)):
            x = (x + np.uint64(SM64_ADD)) & np.uint64(MASK64)
            x = ((x ^ (x >> np.uint64(30))) * np.uint64(SM64_MUL1)) & np.uint64(MASK64)
            x = ((x ^ (x >> np.uint64(27))) * np.uint64(SM64_MUL2)) & np.uint64(MASK64)
            x = x ^ (x >> np.uint64(31))
    return x


def load_workload():
    with h5py.File(DATASET, "r") as f:
        base = np.asarray(f["train_vec"], dtype=np.float32)
        queries = np.asarray(f["test_vec"], dtype=np.float32)
        gt = np.asarray(f["neighbors"], dtype=np.int64)
        train_input = np.asarray(f["train_udf_input"], dtype=np.uint64)
        labels = np.asarray(f["train_label"], dtype=np.int32).reshape(-1)
        rounds = int(f.attrs["udf_rounds"])
        cutoff = int(f.attrs["udf_cutoff"])

    online_values = splitmix64_vector(train_input, rounds)
    online_mask = online_values <= np.uint64(cutoff)
    label_mask = labels == 1
    if not np.array_equal(online_mask, label_mask):
        mismatch = int(np.count_nonzero(online_mask != label_mask))
        raise RuntimeError(f"precomputed label mask mismatch: {mismatch} rows")
    valid_gt = gt[gt >= 0]
    if valid_gt.size and not bool(np.all(label_mask[valid_gt])):
        raise RuntimeError("filtered ground truth contains an object rejected by U1 predicate")
    return base, queries, gt, train_input, label_mask, rounds, cutoff


def load_hnswlib():
    sys.path.insert(0, str(SRC_ROOT))
    import hnswlib  # pylint: disable=import-error

    required = ("reset_metrics", "get_metrics")
    missing = [name for name in required if not hasattr(hnswlib.Index, name)]
    if missing:
        raise RuntimeError(f"instrumented hnswlib missing methods: {missing}")
    return hnswlib


def prepare_index(hnswlib, base: np.ndarray):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index = hnswlib.Index(space="l2", dim=int(base.shape[1]))
    if INDEX_PATH.exists():
        log(f"loading reusable index {INDEX_PATH} ({INDEX_PATH.stat().st_size} bytes)")
        index.load_index(str(INDEX_PATH), max_elements=int(base.shape[0]))
    else:
        log(f"building index M={M}, efConstruction={EF_CONSTRUCTION}, build_threads={BUILD_THREADS}")
        index.init_index(
            max_elements=int(base.shape[0]),
            M=M,
            ef_construction=EF_CONSTRUCTION,
            random_seed=100,
        )
        index.set_num_threads(BUILD_THREADS)
        index.add_items(
            base,
            np.arange(base.shape[0], dtype=np.int64),
            num_threads=BUILD_THREADS,
        )
        index.save_index(str(INDEX_PATH))
        log(f"index saved to {INDEX_PATH} ({INDEX_PATH.stat().st_size} bytes)")
    if int(index.get_current_count()) != int(base.shape[0]):
        raise RuntimeError("index element count does not match U1 base size")
    # Native callback searches are deliberately serial inside hnswlib, while
    # the outer query worker count remains fixed at 28 as in the prior UDF run.
    index.set_num_threads(1)
    index.set_metrics_enabled(False)
    return index


class OnlinePredicate:
    __slots__ = (
        "train_input", "rounds", "cutoff", "timed", "calls", "accepted",
        "rejected", "predicate_ns",
    )

    def __init__(self, train_input, rounds, cutoff, timed=False):
        self.train_input = train_input
        self.rounds = rounds
        self.cutoff = cutoff
        self.timed = timed
        self.calls = 0
        self.accepted = 0
        self.rejected = 0
        self.predicate_ns = 0

    def __call__(self, label):
        start = time.perf_counter_ns() if self.timed else 0
        accepted = splitmix64_scalar(int(self.train_input[int(label)]), self.rounds) <= self.cutoff
        if self.timed:
            self.predicate_ns += time.perf_counter_ns() - start
        self.calls += 1
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return accepted


class PrecomputedPredicate:
    __slots__ = ("mask", "timed", "calls", "accepted", "rejected", "predicate_ns")

    def __init__(self, mask, timed=False):
        self.mask = mask
        self.timed = timed
        self.calls = 0
        self.accepted = 0
        self.rejected = 0
        self.predicate_ns = 0

    def __call__(self, label):
        start = time.perf_counter_ns() if self.timed else 0
        accepted = bool(self.mask[int(label)])
        if self.timed:
            self.predicate_ns += time.perf_counter_ns() - start
        self.calls += 1
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return accepted


def make_fast_callback(mode, mask, train_input, rounds, cutoff):
    if mode == "precomputed":
        def allow(label):
            return bool(mask[int(label)])
    else:
        def allow(label):
            return splitmix64_scalar(int(train_input[int(label)]), rounds) <= cutoff
    return allow


def recall_at_k(ids, truth):
    result = {int(x) for x in ids if int(x) >= 0}
    expected = {int(x) for x in truth if int(x) >= 0}
    return len(result.intersection(expected)) / float(K)


def execute_pass(index, mode, ef, queries, gt, mask, train_input, rounds, cutoff,
                 instrumented, query_count=None):
    n = len(queries) if query_count is None else int(query_count)
    index.set_ef(int(ef))
    index.set_metrics_enabled(bool(instrumented))
    index.reset_metrics()
    records = []
    fast_callback = make_fast_callback(mode, mask, train_input, rounds, cutoff)

    def one(query_id):
        callback = (
            PrecomputedPredicate(mask, timed=True)
            if instrumented and mode == "precomputed"
            else OnlinePredicate(train_input, rounds, cutoff, timed=True)
            if instrumented
            else fast_callback
        )
        start = time.perf_counter_ns()
        ids, _ = index.knn_query(
            queries[query_id:query_id + 1],
            k=K,
            num_threads=1,
            filter=callback,
        )
        elapsed_ns = time.perf_counter_ns() - start
        row = ids[0]
        returned = int(np.count_nonzero(row >= 0))
        recall = recall_at_k(row, gt[query_id])
        result = {
            "query_id": int(query_id),
            "elapsed_ns": int(elapsed_ns),
            "returned": returned,
            "recall": recall,
        }
        if instrumented:
            result.update({
                "predicate_calls": callback.calls,
                "predicate_accepts": callback.accepted,
                "predicate_rejects": callback.rejected,
                "predicate_ns": callback.predicate_ns,
            })
        return result

    start_wall = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=QUERY_WORKERS) as executor:
        records = list(executor.map(one, range(n)))
    wall_ns = time.perf_counter_ns() - start_wall
    native = index.get_metrics()
    elapsed_ms = np.asarray([r["elapsed_ns"] for r in records], dtype=np.float64) / 1e6
    recalls = np.asarray([r["recall"] for r in records], dtype=np.float64)
    returned = np.asarray([r["returned"] for r in records], dtype=np.float64)
    result = {
        "n_queries": n,
        "wall_ms": wall_ns / 1e6,
        "qps": n / (wall_ns / 1e9),
        "latency_mean_ms": float(np.mean(elapsed_ms)),
        "latency_std_ms": float(np.std(elapsed_ms, ddof=1)) if n > 1 else 0.0,
        "latency_p50_ms": float(np.percentile(elapsed_ms, 50)),
        "latency_p95_ms": float(np.percentile(elapsed_ms, 95)),
        "latency_p99_ms": float(np.percentile(elapsed_ms, 99)),
        "recall_mean": float(np.mean(recalls)),
        "returned_mean": float(np.mean(returned)),
        "returned_min": int(np.min(returned)),
        "native_distance_computations": int(native["distance_computations"]),
        "native_hops": int(native["hops"]),
        "native_neighbor_scans": int(native["neighbor_scans"]),
    }
    if instrumented:
        result.update({
            "predicate_calls": int(sum(r["predicate_calls"] for r in records)),
            "predicate_accepts": int(sum(r["predicate_accepts"] for r in records)),
            "predicate_rejects": int(sum(r["predicate_rejects"] for r in records)),
            "predicate_ns": int(sum(r["predicate_ns"] for r in records)),
        })
    return result


def write_record(record):
    with RAW_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed():
    done = set()
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            done.add((obj["phase"], obj["mode"], int(obj["ef"]), int(obj["repeat"])))
    return done


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"starting U1 predicate-cost experiment; data={DATASET}")
    log(f"protocol: M={M}, efConstruction={EF_CONSTRUCTION}, ef={EFS}, k={K}, repeats={REPEATS}, warmup={WARMUP_ROUNDS}x{WARMUP_QUERIES}, query_workers={QUERY_WORKERS}, inner_threads=1")
    base, queries, gt, train_input, mask, rounds, cutoff = load_workload()
    log(f"U1 validated: base={len(base)}, queries={len(queries)}, dim={base.shape[1]}, rounds={rounds}, cutoff={cutoff}, accepted={int(mask.sum())}, selectivity={float(mask.mean()):.8f}")
    hnswlib = load_hnswlib()
    index = prepare_index(hnswlib, base)
    completed = load_completed()

    # A short smoke test is recorded separately and never used as a result.
    if not (RUN_ROOT / "smoke.ok").exists():
        for mode in ("precomputed", "online"):
            smoke = execute_pass(index, mode, 200, queries, gt, mask, train_input, rounds, cutoff, False, 20)
            if smoke["returned_min"] < K:
                raise RuntimeError(f"smoke returned fewer than k for {mode}: {smoke}")
            log(f"smoke {mode}: qps={smoke['qps']:.3f}, recall={smoke['recall_mean']:.6f}, native_dist={smoke['native_distance_computations']}")
        (RUN_ROOT / "smoke.ok").write_text("SMOKE_OK\n", encoding="utf-8")

    for mode in ("precomputed", "online"):
        for ef in EFS:
            log(f"begin mode={mode} ef={ef}")
            # Warmup is identical in shape for both paths and excluded from results.
            for warmup_round in range(WARMUP_ROUNDS):
                warm = execute_pass(index, mode, ef, queries[:WARMUP_QUERIES], gt[:WARMUP_QUERIES], mask, train_input, rounds, cutoff, False)
                log(f"warmup mode={mode} ef={ef} round={warmup_round + 1}/{WARMUP_ROUNDS} qps={warm['qps']:.3f}")

            for repeat in range(1, REPEATS + 1):
                key = ("performance", mode, ef, repeat)
                if key in completed:
                    log(f"skip completed performance mode={mode} ef={ef} repeat={repeat}")
                    continue
                result = execute_pass(index, mode, ef, queries, gt, mask, train_input, rounds, cutoff, False)
                record = {"phase": "performance", "mode": mode, "ef": ef, "repeat": repeat, **result}
                write_record(record)
                completed.add(key)
                log(f"performance mode={mode} ef={ef} repeat={repeat}/{REPEATS} qps={result['qps']:.3f} mean_ms={result['latency_mean_ms']:.3f} p95_ms={result['latency_p95_ms']:.3f} recall={result['recall_mean']:.6f}")

            key = ("counter", mode, ef, 1)
            if key not in completed:
                result = execute_pass(index, mode, ef, queries, gt, mask, train_input, rounds, cutoff, True)
                record = {"phase": "counter", "mode": mode, "ef": ef, "repeat": 1, **result}
                write_record(record)
                completed.add(key)
                log(f"counter mode={mode} ef={ef} calls/q={result['predicate_calls']/result['n_queries']:.2f} accept={result['predicate_accepts']/result['n_queries']:.2f} reject={result['predicate_rejects']/result['n_queries']:.2f} pred_us/q={result['predicate_ns']/result['n_queries']/1000:.2f} dist/q={result['native_distance_computations']/result['n_queries']:.2f} scans/q={result['native_neighbor_scans']/result['n_queries']:.2f}")

    records = [json.loads(x) for x in RAW_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    performance = [r for r in records if r["phase"] == "performance"]
    counters = [r for r in records if r["phase"] == "counter"]
    if len(performance) != len(EFS) * 2 * REPEATS or len(counters) != len(EFS) * 2:
        log(f"incomplete: performance={len(performance)}, counters={len(counters)}")
        return 2
    summary = {
        "status": "complete",
        "dataset": str(DATASET),
        "index": str(INDEX_PATH),
        "base": int(len(base)),
        "queries": int(len(queries)),
        "dim": int(base.shape[1]),
        "k": K,
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "efs": list(EFS),
        "repeats": REPEATS,
        "warmup_rounds": WARMUP_ROUNDS,
        "warmup_queries": WARMUP_QUERIES,
        "query_workers": QUERY_WORKERS,
        "inner_threads": 1,
        "udf_rounds": rounds,
        "udf_cutoff": cutoff,
        "accepted": int(mask.sum()),
        "selectivity": float(mask.mean()),
        "performance_records": len(performance),
        "counter_records": len(counters),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RUN_ROOT / "EXPERIMENT_COMPLETE").write_text("EXPERIMENT_COMPLETE\n", encoding="utf-8")
    log("EXPERIMENT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
