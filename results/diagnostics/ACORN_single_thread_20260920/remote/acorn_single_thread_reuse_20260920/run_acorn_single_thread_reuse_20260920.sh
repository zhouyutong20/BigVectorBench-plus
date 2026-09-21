#!/usr/bin/env bash
set -u

RUN_DIR=<remote_artifact>
BIN=<remote_artifact>
LIBS=<remote_artifact>
INDEX_ROOT=<remote_artifact>

datasets=(
  artificial-corr-12-original
  artificial-corr-12-shuffled
  cc_news-original
  cc_news-shuffled
)
ns=(991000 991000 620643 620643)
cpus=(0 28 56 84)

mkdir -p "$RUN_DIR/logs"

pids=()
for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  n="${ns[$i]}"
  cpu="${cpus[$i]}"
  work="$INDEX_ROOT/$dataset/acorn/work"
  log="$RUN_DIR/logs/$dataset.log"
  (
    cd "$work" || exit 20
    export LD_LIBRARY_PATH="$LIBS"
    export OMP_NUM_THREADS=1
    export ACORN_BUILD_THREADS=1
    export ACORN_SEARCH_THREADS=1
    export ACORN_FILTER_THREADS=1
    export ACORN_EFSEARCH_LIST=100,200,400,800,1600
    export BVB_WARMUP_RUNS=2
    export BVB_WARMUP_QUERIES=1000
    taskset -c "$cpu" "$BIN" "$n" 12 "$dataset" 64 64 400 100,200,400,800,1600 >"$log" 2>&1
    rc=$?
    echo "RUN_RC=$rc" >>"$log"
    exit "$rc"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -eq 0 ]]; then
  date -Is >"$RUN_DIR/ACORN_SINGLE_THREAD_REUSE_COMPLETE"
  exit 0
fi

date -Is >"$RUN_DIR/ACORN_SINGLE_THREAD_REUSE_FAILED"
exit 1
