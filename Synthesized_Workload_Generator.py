# -*- coding: utf-8 -*-
# NOTE: English identifiers only; Chinese comments allowed; comments use '#'

import os
import argparse
from typing import Tuple, List

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# ---------------- argparse helpers ----------------
def positive_int(input_str: str) -> int:
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer") from exc
    return i


def positive_float(input_str: str) -> float:
    try:
        x = float(input_str)
        if x <= 0:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive float") from exc
    return x


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data source
    parser.add_argument(
        "--source",
        default="hdf5",
        choices={"hdf5", "blobs", "gaussian", "uniform"},
        help="vector source: read from hdf5 or synthesize vectors",
    )
    parser.add_argument(
        "--in_fn",
        default="",
        help="input hdf5 containing datasets 'train' and 'test' (required if source=hdf5)",
    )
    parser.add_argument("--out_fn", default="out_filter_ann.hdf5", help="output hdf5 path")

    # sizes / dims
    parser.add_argument("--n_samples", default=100000, type=positive_int, help="number of train vectors")
    parser.add_argument("--m_test", default=10000, type=positive_int, help="number of test vectors")
    parser.add_argument("--n_dims", default=128, type=positive_int, help="vector dimension")
    parser.add_argument("--centers", default=100, type=positive_int, help="number of centers (for blobs)")

    # filter params
    parser.add_argument("--n_filters", default=1, type=positive_int, help="number of filter dimensions")
    parser.add_argument("--max_labels", default=100000, type=positive_int, help="max label value")
    parser.add_argument("--ratio_request", default=0.03, type=float, help="requested overall selectivity ratio (static mode)")

    # knn params
    parser.add_argument("--distance", default="euclidean", choices={"euclidean", "angular", "inner_product"}, help="distance metric")
    parser.add_argument("--topk", default=200, type=positive_int, help="topk neighbors")

    # range generation mode
    parser.add_argument("--mode", default="static", choices={"static", "dynamic_beta"}, help="range generation mode")
    parser.add_argument("--mean_ratio", default=0.03, type=float, help="mean ratio for dynamic_beta")
    parser.add_argument("--min_ratio", default=0.001, type=float, help="min ratio for dynamic_beta")
    parser.add_argument("--max_ratio", default=0.25, type=float, help="max ratio for dynamic_beta")
    parser.add_argument("--beta_shape", default=2.0, type=positive_float, help="beta shape for dynamic_beta")

    args = parser.parse_args()
    return args


# ---------------- metrics ----------------
def inner_product_metric(u: np.ndarray, v: np.ndarray) -> float:
    # sklearn.NearestNeighbors custom metric: larger dot -> smaller "distance", so use negative dot
    return -float(np.dot(u, v))


def metric_mapping(metric: str):
    metric = metric.lower()
    metric_type = {"angular": "cosine", "euclidean": "euclidean", "inner_product": inner_product_metric}.get(metric, None)
    if metric_type is None:
        raise ValueError(f"unsupported metric: {metric}")
    return metric_type


# ---------------- KNN + filter（保持源代码逻辑） ----------------
def filter_knn_calculate(
    distance: str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    topk: int,
    ratio_request: float,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbors_ds = np.full((len(test_vec), topk), -1, dtype=np.int32)
    distances_ds = np.full((len(test_vec), topk), -1.0, dtype=np.float32)
    maxcnt = 0
    mincnt = train_vec.shape[0]
    sumcnt = 0

    if ratio_request >= 0.2:
        # Branch A: KNN on full set then post-filter
        for i, qry in tqdm(enumerate(test_vec), desc="Processing"):
            vec_set, idx_set = [], []

            n_samples_fit = min(
                int(max(topk * 1.5 / max(ratio_request, 1e-6), topk)),
                train_vec.shape[0],
            )
            nn = NearestNeighbors(
                n_neighbors=n_samples_fit,
                metric=metric_mapping(distance),
                n_jobs=-1,
                algorithm="brute",
            )
            nn.fit(train_vec)
            distances, indices = nn.kneighbors(np.array([qry]))

            neighbors_tmp = np.full((n_samples_fit,), -1, dtype=np.int32)
            distances_tmp = np.full((n_samples_fit,), -1.0, dtype=np.float32)
            labels_tmp = np.full((n_samples_fit, train_label.shape[1]), 0, dtype=np.int32)

            neighbors_tmp[:n_samples_fit] = indices[0]
            distances_tmp[:n_samples_fit] = distances[0]
            labels_tmp[:n_samples_fit] = train_label[indices[0]]

            flags = np.zeros(n_samples_fit, dtype=int)
            for j, j_test_label in enumerate(test_label[i]):
                left, right = j_test_label[0], j_test_label[1]
                flags += (left <= labels_tmp[:, j]) & (labels_tmp[:, j] <= right)

            cnt = 0
            lens = train_label.shape[1]
            for j, flag in enumerate(flags):
                if flag == lens:
                    vec_set.append(distances_tmp[j])
                    idx_set.append(neighbors_tmp[j])
                    cnt += 1

            if cnt < topk:
                print(f"Warning: {i}-th query has {cnt} neighbors")

            dist_scope = np.array(vec_set, dtype=np.float32)
            idx_scope = np.array(idx_set, dtype=np.int32)

            if i % 100 == 0:
                print(f"{i}-th query : dist_scope.shape: {dist_scope.shape}, idx_scope.shape: {idx_scope.shape}")

            n_fill = min(topk, dist_scope.shape[0])
            if n_fill < 1:
                print(f"Warning: {i}-th query has {n_fill} neighbors")
                continue

            maxcnt = max(maxcnt, cnt)
            mincnt = min(mincnt, cnt)
            sumcnt += cnt

            neighbors_ds[i, :n_fill] = idx_scope[:n_fill]
            distances_ds[i, :n_fill] = dist_scope[:n_fill]

        denom = int(max(topk * 2 / max(ratio_request, 1e-6), 1))
        print(f"min filter ratio is {mincnt / denom}")
        print(f"max filter ratio is {maxcnt / denom}")
        print(f"average filter ratio is {sumcnt / (denom * test_vec.shape[0])}")

    else:
        # Branch B: pre-filter then KNN on subset
        for i, qry in tqdm(enumerate(test_vec), desc="Processing"):
            vec_set, idx_set = [], []
            lens = train_label.shape[1]

            flags = np.zeros(train_label.shape[0], dtype=int)
            for j, j_test_label in enumerate(test_label[i]):
                left, right = j_test_label[0], j_test_label[1]
                flags += (left <= train_label[:, j]) & (train_label[:, j] <= right)

            cnt = 0
            for j, flag in enumerate(flags):
                if flag == lens:
                    vec_set.append(train_vec[j])
                    idx_set.append(j)
                    cnt += 1

            if cnt < topk:
                print(f"Warning: {i}-th query has {cnt} neighbors")

            train_vec_scope = np.array(vec_set, dtype=np.float32)
            train_idx_scope = np.array(idx_set, dtype=np.int32)

            if i % 100 == 0:
                print(
                    f"{i}-th query : train_vec_scope.shape: {train_vec_scope.shape}, "
                    f"train_idx_scope.shape: {train_idx_scope.shape}"
                )

            n_samples_fit = min(topk, train_vec_scope.shape[0])
            if n_samples_fit < 1:
                print(f"Warning: {i}-th query has {n_samples_fit} neighbors")
                continue

            maxcnt = max(maxcnt, cnt)
            mincnt = min(mincnt, cnt)
            sumcnt += cnt

            nn = NearestNeighbors(
                n_neighbors=n_samples_fit,
                metric=metric_mapping(distance),
                n_jobs=-1,
                algorithm="brute",
            )
            nn.fit(train_vec_scope)
            distances, indices = nn.kneighbors(np.array([qry]))

            neighbors_ds[i, :n_samples_fit] = train_idx_scope[indices[0]]
            distances_ds[i, :n_samples_fit] = distances[0]

        print(f"min filter ratio is {mincnt / (train_vec.shape[0])}")
        print(f"max filter ratio is {maxcnt / (train_vec.shape[0])}")
        print(f"average filter ratio is {sumcnt / (train_vec.shape[0] * test_vec.shape[0])}")

    if distance == "inner_product":
        distances_ds = -distances_ds
    return neighbors_ds, distances_ds


# ---------------- range generation: STATIC / DYNAMIC_BETA（保持源代码逻辑） ----------------
def _sample_ratio_beta(
    mean_ratio: float,
    min_ratio: float,
    max_ratio: float,
    shape: float,
    rng: np.random.Generator,
) -> float:
    eps = 1e-12
    if not (0.0 < min_ratio < max_ratio <= 1.0):
        raise ValueError("min_ratio/max_ratio must satisfy 0 < min_ratio < max_ratio <= 1")
    mean_ratio = float(np.clip(mean_ratio, min_ratio + eps, max_ratio - eps))
    shape = float(shape)
    if shape <= 0:
        raise ValueError("beta shape must be > 0")

    mean01 = (mean_ratio - min_ratio) / (max_ratio - min_ratio)
    mean01 = float(np.clip(mean01, eps, 1.0 - eps))
    alpha = shape
    beta = alpha * (1.0 - mean01) / mean01

    x = rng.gamma(shape=alpha, scale=1.0)
    y = rng.gamma(shape=beta, scale=1.0)
    r01 = x / (x + y)
    r = min_ratio + (max_ratio - min_ratio) * r01
    return float(r)


def generate_random_ranges(
    generate_type: str,
    total: int,
    num_ranges: int,
    min_val: int,
    max_val: int,
    train_labels: np.ndarray,
    ratio_request: float,
    *,
    mode: str = "static",
    mean_ratio: float = 0.03,
    min_ratio: float = 0.001,
    max_ratio: float = 0.25,
    beta_shape: float = 2.0,
):
    # 中文注释：严格保持你新脚本的两种模式
    rng = np.random.default_rng(42)
    l = int(num_ranges)
    if l <= 0:
        raise ValueError("num_ranges must be >= 1")
    if not (0 <= min_val <= max_val):
        raise ValueError("min_val/max_val invalid")

    total_span = float(max_val - min_val + 1)
    label_ranges: List[List[Tuple[int, int]]] = []

    if mode == "dynamic_beta":
        avg_ratio = 0.0
        for _ in tqdm(range(total), desc="Processing"):
            ratio = _sample_ratio_beta(mean_ratio, min_ratio, max_ratio, beta_shape, rng)
            avg_ratio += ratio
            per_dim_ratio = ratio if l == 1 else ratio ** (1.0 / l)

            ranges_i: List[Tuple[int, int]] = []
            for _d in range(l):
                sub_len = max(1.0, per_dim_ratio * (total_span - 1))
                start_max = max_val - sub_len
                if start_max <= min_val:
                    left = float(min_val)
                else:
                    left = float(rng.uniform(min_val, start_max))
                right = left + sub_len

                left_i = int(max(min_val, min(right, left)))
                right_i = int(min(max_val, max(left, right)))
                if right_i <= left_i:
                    right_i = min(int(left_i + 1), int(max_val))
                ranges_i.append((left_i, right_i))
            label_ranges.append(ranges_i)

        print(f"average ratio is: {avg_ratio / max(1, total)}")
        return label_ranges

    if mode == "static":
        if not (0.0 < ratio_request <= 1.0):
            raise ValueError("static ratio_request must satisfy 0 < ratio_request <= 1")

        per_dim_ratio = float(ratio_request) if l == 1 else float(ratio_request) ** (1.0 / l)
        sub_len = max(1, int(round(per_dim_ratio * (total_span - 1))))

        for _ in tqdm(range(total), desc="Processing"):
            ranges_i: List[Tuple[int, int]] = []
            for _d in range(l):
                start_max = max_val - sub_len
                if start_max <= min_val:
                    left_i = int(min_val)
                else:
                    left_i = int(rng.integers(min_val, start_max + 1))
                right_i = int(min(max_val, left_i + sub_len))
                if right_i <= left_i:
                    right_i = min(left_i + 1, int(max_val))
                ranges_i.append((left_i, right_i))
            label_ranges.append(ranges_i)
        return label_ranges

    raise ValueError("mode must be 'static' or 'dynamic_beta'")


# ---------------- I/O helpers（保持源代码逻辑） ----------------
def write_filter_output(
    fn: str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    distance: str,
    filter_expr_func: str,
    label_names: List[str],
    label_types: List[str],
    label_ranges: List[str],
    label_range_types: List[str],
    point_type: str,
    topk: int,
    ratio_request: float,
    mode: str = "static",
    mean_ratio: float = 0.03,
    min_ratio: float = 0.001,
    max_ratio: float = 0.25,
    beta_shape: float = 2.0,
) -> None:
    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "filter-ann"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = int(train_vec.shape[1])
        f.attrs["point_type"] = point_type
        f.attrs["label_names"] = label_names
        f.attrs["label_types"] = label_types
        f.attrs["label_ranges"] = label_ranges
        f.attrs["label_range_types"] = label_range_types
        f.attrs["filter_expr_func"] = filter_expr_func

        # record range generation strategy
        f.attrs["range_mode"] = mode
        f.attrs["range_ratio_request"] = float(ratio_request)
        f.attrs["range_mean_ratio"] = float(mean_ratio)
        f.attrs["range_min_ratio"] = float(min_ratio)
        f.attrs["range_max_ratio"] = float(max_ratio)
        f.attrs["range_beta_shape"] = float(beta_shape)

        print(f"train size: {train_vec.shape[0]} * {train_vec.shape[1]}")
        print(f"test size:  {test_vec.shape[0]} * {test_vec.shape[1]}")

        f.create_dataset(
            "train_vec",
            data=train_vec,
            maxshape=(None, train_vec.shape[1]),
            chunks=(10000, train_vec.shape[1]),
            dtype=float,
        )
        f.create_dataset(
            "test_vec",
            data=test_vec,
            maxshape=(None, test_vec.shape[1]),
            chunks=(10000, test_vec.shape[1]),
            dtype=float,
        )

        test_label_scope = np.array(test_label, dtype=np.int32)
        print(f"train_label size: {train_label.shape[0]} * {train_label.shape[1]}")
        print(f"test_label size:  {test_label_scope.shape[0]} * {test_label_scope.shape[1]} * {test_label_scope.shape[2]}")

        f.create_dataset(
            "train_label",
            data=train_label,
            maxshape=(None, train_label.shape[1]),
            chunks=(10000, train_label.shape[1]),
            dtype=int,
        )
        f.create_dataset(
            "test_label",
            data=test_label_scope,
            maxshape=(None, None, test_label_scope.shape[2]),
            chunks=(10000, test_label_scope.shape[1], test_label_scope.shape[2]),
            dtype=int,
        )

        neighbors, distances = filter_knn_calculate(
            distance, train_vec, test_vec, train_label, test_label, topk, ratio_request
        )

        f.create_dataset(
            "neighbors",
            data=neighbors,
            maxshape=(None, neighbors.shape[1]),
            chunks=(10000, neighbors.shape[1]),
            dtype=int,
        )
        f.create_dataset(
            "distances",
            data=distances,
            maxshape=(None, distances.shape[1]),
            chunks=(10000, distances.shape[1]),
            dtype=float,
        )

    print(f"datafile is already: {fn}")


# ---------------- vector source: strict synth (required) ----------------
def _strict_train_test_split(x: np.ndarray, m_test: int) -> Tuple[np.ndarray, np.ndarray]:
    # 中文注释：严格复刻旧脚本逻辑：train_test_split(test_size=m_test, random_state=42)
    n_total = int(x.shape[0])
    if m_test <= 0 or m_test >= n_total:
        raise ValueError(f"m_test must satisfy 0 < m_test < n_total, got m_test={m_test}, n_total={n_total}")

    train_x, test_x = train_test_split(
        x,
        test_size=int(m_test),
        random_state=42,
        shuffle=True,
    )
    return train_x, test_x


def _synthesize_vectors(
    source: str,
    n_samples: int,
    m_test: int,
    n_dims: int,
    centers: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    # 中文注释：严格版——先生成 n_samples+m_test，再用 train_test_split(test_size=m_test, random_state=42) 切分
    n_total = int(n_samples + m_test)

    if source == "blobs":
        x, _ = make_blobs(
            n_samples=n_total,
            n_features=int(n_dims),
            centers=int(centers),
            random_state=1,  # 中文注释：严格沿用旧脚本 make_blobs(random_state=1)
        )
        x = x.astype(np.float32, copy=False)
        train_x, test_x = _strict_train_test_split(x, m_test)
        return train_x, test_x

    if source == "gaussian":
        x = rng.standard_normal(size=(n_total, n_dims)).astype(np.float32, copy=False)
        train_x, test_x = _strict_train_test_split(x, m_test)
        return train_x, test_x

    if source == "uniform":
        x = rng.uniform(low=-1.0, high=1.0, size=(n_total, n_dims)).astype(np.float32, copy=False)
        train_x, test_x = _strict_train_test_split(x, m_test)
        return train_x, test_x

    raise ValueError(f"unsupported synth source: {source}")


def _load_vectors_from_hdf5(
    in_fn: str,
    n_samples: int,
    m_test: int,
    n_dims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # 中文注释：保持你新脚本的“固定规模采样 + 不足用 train 补齐”逻辑
    with h5py.File(in_fn, "r") as f:
        train_dset = f["train"]
        test_dset = f["test"]

        if train_dset.shape[1] != n_dims:
            print(f"[warn] train dim={train_dset.shape[1]} != n_dims={n_dims}")
        if test_dset.shape[1] != n_dims:
            print(f"[warn] test  dim={test_dset.shape[1]} != n_dims={n_dims}")

        target_train = min(int(n_samples), int(train_dset.shape[0]))
        target_test = min(int(m_test), int(test_dset.shape[0]))

        rng = np.random.default_rng(42)

        # train: sample target_train
        train_idx = np.sort(rng.choice(train_dset.shape[0], size=target_train, replace=False))
        train_x = train_dset[train_idx].astype(np.float32, copy=False)

        # test: sample from test if enough, else pad with train excluding train_idx
        base = int(m_test) if target_test >= int(m_test) else target_test
        if base > 0:
            test_idx = np.sort(rng.choice(test_dset.shape[0], size=base, replace=False))
            test_x = test_dset[test_idx].astype(np.float32, copy=False)
        else:
            test_x = np.empty((0, test_dset.shape[1]), dtype=np.float32)

        need = int(m_test) - test_x.shape[0]
        if need > 0:
            all_train_idx = np.arange(train_dset.shape[0], dtype=np.int64)
            mask = np.ones(train_dset.shape[0], dtype=bool)
            mask[train_idx] = False
            avail = all_train_idx[mask]
            extra_idx = np.sort(rng.choice(avail, size=need, replace=False))
            extra_x = train_dset[extra_idx].astype(np.float32, copy=False)
            test_x = np.vstack([test_x, extra_x])

    return train_x, test_x


# ---------------- dataset creation (same logic; only vector source extended) ----------------
def create_filter(
    in_fn: str,
    out_fn: str,
    generate_type: str,
    n_dims: int,
    n_samples: int,
    m_test: int,
    centers: int,
    n_filters: int,
    max_labels: int,
    ratio_request: float,
    distance: str = "inner_product",
    topk: int = 200,
    mode: str = "static",
    mean_ratio: float = 0.03,
    min_ratio: float = 0.001,
    max_ratio: float = 0.25,
    beta_shape: float = 2.0,
    source: str = "hdf5",
) -> None:
    # 中文注释：除“向量来源”外，其余结构与源代码一致
    print(f"now_dataset: {in_fn if source == 'hdf5' else source}")

    rng = np.random.default_rng(42)

    if source == "hdf5":
        if not in_fn:
            raise ValueError("in_fn is required when source='hdf5'")
        train_x, test_x = _load_vectors_from_hdf5(in_fn, n_samples, m_test, n_dims)
    else:
        train_x, test_x = _synthesize_vectors(source, n_samples, m_test, n_dims, centers, rng)

    print(f"train_X shape: {train_x.shape}  test_X shape: {test_x.shape}")

    # labels and filter expr
    train_label_names = [f"label_{i}" for i in range(n_filters)]
    train_label_types = ["int32" for _ in range(n_filters)]
    print(f"train_labels_names: {train_label_names}")
    print(f"train_labels_types: {train_label_types}")

    train_label_min = 1
    train_label_max = max_labels

    cols = []
    for _ in range(n_filters):
        cols.append(np.random.randint(train_label_min, train_label_max, size=train_x.shape[0], dtype=np.int32))
    train_labels = np.stack(cols, axis=1)
    print(train_labels.shape)

    test_label_range_names = [[f"label_l_{i}", f"label_r_{i}"] for i in range(n_filters)]
    test_label_range_types = [["int32", "int32"] for _ in range(n_filters)]
    print(f"test_label_range_names: {test_label_range_names}")
    print(f"test_label_range_types: {test_label_range_types}")

    if n_filters > 1:
        filter_expr_func = (
            "def filter_expr("
            + ",".join([f"label_l_{i}, label_r_{i}" for i in range(n_filters)])
            + "):\n"
            + '    return " and ".join(['
            + " , ".join([f'f"label_{i} >= {{label_l_{i}}} and label_{i} <= {{label_r_{i}}}"' for i in range(n_filters)])
            + "])\n"
        )
    else:
        filter_expr_func = (
            "def filter_expr(label_l_0, label_r_0):\n"
            "    return f\"label_0 >= {label_l_0} and label_0 <= {label_r_0}\"\n"
        )

    print(f"filter_expr_func: {filter_expr_func}")

    # generate ranges
    test_label_min = 0
    test_label_max = max_labels
    test_labels_range = generate_random_ranges(
        generate_type,
        test_x.shape[0],
        n_filters,
        test_label_min,
        test_label_max,
        train_labels,
        ratio_request,
        mode=mode,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        beta_shape=beta_shape,
    )

    # write output + compute neighbors/distances
    write_filter_output(
        out_fn,
        train_x,
        test_x,
        train_labels,
        test_labels_range,
        distance,
        filter_expr_func,
        train_label_names,
        train_label_types,
        test_label_range_names,
        test_label_range_types,
        "float",
        topk,
        ratio_request,
        mode=mode,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        beta_shape=beta_shape,
    )


# ---------------- main ----------------
if __name__ == "__main__":
    args = parse_arguments()

    create_filter(
        in_fn=args.in_fn,
        out_fn=args.out_fn,
        generate_type="random",
        n_dims=args.n_dims,
        n_samples=args.n_samples,
        m_test=args.m_test,
        centers=args.centers,
        n_filters=args.n_filters,
        max_labels=args.max_labels,
        ratio_request=args.ratio_request,
        distance=args.distance,
        topk=args.topk,
        mode=args.mode,
        mean_ratio=args.mean_ratio,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        beta_shape=args.beta_shape,
        source=args.source,
    )

