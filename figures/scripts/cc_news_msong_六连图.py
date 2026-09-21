"""Draw a 2 x 3 Recall-QPS figure for cc_news and msong.

The three columns follow the existing plotting scripts in this directory:
Algorithms, Libraries, and Databases.  The first row is cc_news and the
second row is msong.
"""

from __future__ import annotations

import glob
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PNG = SCRIPT_DIR / "cc_news_msong_6panel.png"
OUTPUT_PDF = SCRIPT_DIR / "cc_news_msong_6panel.pdf"
AUDIT_CSV = SCRIPT_DIR / "cc_news_msong_6panel_audit.csv"

DATASETS = ["cc_news", "msong"]
CATEGORIES = ["Algorithms", "Libraries", "Databases"]

# Keep the whitelist used by the current 二连图.py.  It avoids mixing in
# unrelated old baselines while retaining the current CAGRA measurements.
TARGET_ALGORITHMS = {
    "rangepq",
    "digra",
    "acorn",
    "irangegraph",
    "hnsw+post",
    "hnsw",
    "ivfpq",
    "milvus-hnsw",
    "milvus-ivfpq",
    "milvus-gpu-ivfpq",
    "milvus-gpu-cagra",
    "weaviate-hnsw",
    "qdrant",
    "vearch-ivfpq",
    "vearch-hnsw",
    "redis-hnsw",
    "elasticsearch-hnsw",
    "pgvector-hnsw",
    "digra_w1-4_s1-6",
    "faiss-hnsw",
    "faiss-ivfpq",
    "hnswlib",
    "milvus-gpu-ivfflat",
}

COLORS = {
    # A more separated palette keeps neighboring legend entries readable.
    "acorn": "#17becf",
    "digra": "#756bb1",
    "irangegraph": "#d62728",
    "iRangeGraph": "#d62728",
    "rangepq": "#e377c2",
    "Faiss-HNSW": "#1f77b4",
    "Faiss-IVFPQ": "#2ca02c",
    "HNSWLib": "#ff7f0e",
    "elasticsearch-hnsw": "#8c564b",
    "milvus-gpu-cagra": "#4c78a8",
    "milvus-gpu-ivfflat": "#59a14f",
    "milvus-gpu-ivfpq": "#2a9d8f",
    "milvus-hnsw": "#f28e2b",
    "milvus-ivfpq": "#ff9da7",
    "pgvector-hnsw": "#76b041",
    "qdrant": "#b279a2",
    "redis-hnsw": "#9c755f",
    "vearch-hnsw": "#edc948",
    "vearch-ivfpq": "#af7aa1",
    "weaviate-hnsw": "#e15759",
    "results": "#9467bd",
}

DISPLAY_NAMES = {
    "acorn": "ACORN",
    "digra": "DiGRA",
    "irangegraph": "iRangeGraph",
    "rangepq": "RangePQ",
    "Faiss-HNSW": "Faiss-HNSW",
    "Faiss-IVFPQ": "Faiss-IVFPQ",
    "HNSWLib": "HNSWlib-HNSW",
    "milvus-gpu-ivfflat": "Milvus-GPU-IVFFLAT",
    "milvus-gpu-cagra": "Milvus-GPU-CAGRA",
    "milvus-gpu-ivfpq": "Milvus-GPU-IVFPQ",
    "milvus-hnsw": "Milvus-HNSW",
    "milvus-ivfpq": "Milvus-IVFPQ",
    "pgvector-hnsw": "PGVector-HNSW",
    "qdrant": "Qdrant",
    "elasticsearch-hnsw": "Elasticsearch-HNSW",
    "weaviate-hnsw": "Weaviate-HNSW",
    "redis-hnsw": "Redis-HNSW",
    "vearch-hnsw": "Vearch-HNSW",
    "vearch-ivfpq": "Vearch-IVFPQ",
    "results": "Results",
}

MARKERS = {
    "acorn": "X",
    "digra": "v",
    "rangepq": "D",
    "milvus-gpu-ivfflat": "v",
    "milvus-gpu-cagra": "h",
    "milvus-gpu-ivfpq": "p",
    "milvus-hnsw": "P",
    "milvus-ivfpq": "d",
    "pgvector-hnsw": "+",
    "qdrant": "o",
    "elasticsearch-hnsw": "P",
    "weaviate-hnsw": "s",
    "redis-hnsw": "*",
    "vearch-hnsw": "^",
    "vearch-ivfpq": "H",
    "irangegraph": "o",
    "iRangeGraph": "o",
    "Faiss-HNSW": "s",
    "Faiss-IVFPQ": "D",
    "HNSWLib": "^",
    "results": "X",
}

FALLBACK_MARKERS = ["o", "v", "^", "s", "p", "*", "h", "H", "D", "d", "P", "X", "+"]
FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

CAT_DB = ["milvus", "weaviate", "qdrant", "vearch", "redis", "elastic", "pgvector"]
CAT_LIB = ["faiss", "hnswlib", "hnsw+post"]


def normalize_dataset_name(name: object) -> str:
    value = str(name).lower()
    if "cc_news" in value:
        return "cc_news"
    if "msong" in value:
        return "msong"
    return value


def infer_algorithm(file_name: str) -> str:
    lower = file_name.lower()
    if "irangegraph" in lower:
        return "iRangeGraph"
    if lower.startswith("digra"):
        return "digra"
    if lower.startswith("rangepq"):
        return "rangepq"
    if lower.startswith("acorn"):
        return "acorn"
    if lower == "results.pkl":
        return "results"
    return Path(file_name).stem


def add_frame_records(records: list[dict], frame: pd.DataFrame, file_name: str) -> int:
    if frame is None or frame.empty:
        return 0

    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame.rename(
        columns={"k-nn": "recall", "knn": "recall", "recall@1": "recall"},
        inplace=True,
    )
    required = {"dataset", "recall", "qps"}
    if not required.issubset(frame.columns):
        return 0

    if "algorithm" not in frame.columns:
        frame["algorithm"] = infer_algorithm(file_name)

    frame["recall"] = pd.to_numeric(frame["recall"], errors="coerce")
    frame["qps"] = pd.to_numeric(frame["qps"], errors="coerce")
    frame = frame[["algorithm", "dataset", "recall", "qps"]].dropna()
    if frame.empty:
        return 0

    records.extend(frame.to_dict("records"))
    return len(frame)


def read_data_files() -> list[dict]:
    records: list[dict] = []
    paths = sorted(SCRIPT_DIR.glob("*.csv")) + sorted(SCRIPT_DIR.glob("*.pkl")) + sorted(SCRIPT_DIR.glob("*.xlsx"))

    for path in paths:
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                count = add_frame_records(records, frame, path.name)
            elif path.suffix.lower() == ".xlsx":
                frame = pd.read_excel(path)
                count = add_frame_records(records, frame, path.name)
            else:
                raw = pd.read_pickle(path)
                if isinstance(raw, dict) and "dataset" not in raw:
                    temp: list[dict] = []
                    algorithm = infer_algorithm(path.name)
                    for dataset, points in raw.items():
                        if not isinstance(points, (list, tuple)):
                            continue
                        for point in points:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                temp.append(
                                    {
                                        "algorithm": algorithm,
                                        "dataset": dataset,
                                        "recall": point[0],
                                        "qps": point[1],
                                    }
                                )
                    count = add_frame_records(records, pd.DataFrame(temp), path.name)
                else:
                    count = add_frame_records(records, raw, path.name)

            if count:
                print(f"[data] {path.name}: {count} rows")
        except Exception as error:
            print(f"[skip] {path.name}: {error}")

    return records


def parse_standard_log(path: Path, algorithm: str) -> list[dict]:
    parsed: list[dict] = []
    content = path.read_text(encoding="utf-8", errors="ignore")
    for line in content.splitlines():
        if "[RESULT]" not in line:
            continue
        dataset = re.search(r"dataset=([A-Za-z0-9_-]+)", line)
        recall = re.search(r"(?:recall|avg_recall@\d+)=([0-9.]+)", line)
        qps = re.search(r"qps=([0-9.]+)", line)
        if dataset and recall and qps:
            parsed.append(
                {
                    "algorithm": algorithm,
                    "dataset": dataset.group(1),
                    "recall": float(recall.group(1)),
                    "qps": float(qps.group(1)),
                }
            )
    return parsed


def parse_acorn_log(path: Path) -> list[dict]:
    parsed: list[dict] = []
    content = path.read_text(encoding="utf-8", errors="ignore")
    dataset_match = re.search(r"dataset\s*[:=]\s*([A-Za-z0-9_-]+)", content, re.IGNORECASE)
    if not dataset_match:
        return parsed
    dataset = dataset_match.group(1)
    for line in content.splitlines():
        qps = re.search(r"QPS\s*=\s*([0-9.]+)", line)
        recall = re.search(r"(?:recall|avg_recall)@\d+\s*=\s*([0-9.]+)", line)
        if qps and recall:
            parsed.append(
                {
                    "algorithm": "acorn",
                    "dataset": dataset,
                    "recall": float(recall.group(1)),
                    "qps": float(qps.group(1)),
                }
            )
    return parsed


def read_log_files(records: list[dict]) -> None:
    # Explicitly map the logs so the duplicated hnsw_post.log is not counted
    # twice as two differently named library implementations.
    standard_specs = [
        ("Faiss-HNSW", ["faiss_hnsw*.log", "faiss_hnsw_151"]),
        ("Faiss-IVFPQ", ["faiss_ivfpq*.log", "faiss_ivfpq_151"]),
        ("HNSWLib", ["hnswlib*.log", "bvbj_hnswlib_151"]),
    ]
    seen: set[Path] = set()
    for algorithm, patterns in standard_specs:
        for pattern in patterns:
            for path in sorted(SCRIPT_DIR.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                parsed = parse_standard_log(path, algorithm)
                if parsed:
                    records.extend(parsed)
                    seen.add(path)
                    print(f"[log] {path.name}: {len(parsed)} rows as {algorithm}")

    # ACORN logs have QPS=..., recall@... lines rather than [RESULT] lines.
    acorn_paths: set[Path] = set()
    for pattern in ["*gamma*.log", "*acorn*.log", "cc_news*.log", "msong*.log", "output_*.log"]:
        acorn_paths.update(path for path in SCRIPT_DIR.glob(pattern) if path.is_file())
    for path in sorted(acorn_paths):
        parsed = parse_acorn_log(path)
        if parsed:
            records.extend(parsed)
            print(f"[log] {path.name}: {len(parsed)} rows as acorn")


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = read_data_files()
    read_log_files(records)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No readable data rows were found.")

    frame["dataset"] = frame["dataset"].map(normalize_dataset_name)
    frame["algorithm"] = frame["algorithm"].map(
        lambda value: "HNSWLib" if str(value).lower() == "hnsw+post" else str(value)
    )
    frame["algorithm_key"] = frame["algorithm"].str.lower()
    frame["recall"] = pd.to_numeric(frame["recall"], errors="coerce")
    frame["qps"] = pd.to_numeric(frame["qps"], errors="coerce")
    raw_frame = frame[
        frame["dataset"].isin(DATASETS)
        & frame["recall"].notna()
        & frame["qps"].notna()
        & (frame["recall"] >= 0)
        & (frame["qps"] > 0)
    ].copy()
    raw_frame["category"] = raw_frame["algorithm"].map(get_category)

    frame = raw_frame[raw_frame["algorithm_key"].isin(TARGET_ALGORITHMS)].copy()
    frame = frame.drop(columns=["algorithm_key"])
    raw_frame = raw_frame.drop(columns=["algorithm_key"])
    if frame.empty:
        raise RuntimeError("No cc_news/msong rows remain after dataset and algorithm filtering.")

    print("[summary]")
    print(frame.groupby(["dataset", "algorithm"]).size().to_string())
    return frame, raw_frame


def get_category(algorithm: str) -> str:
    lower = algorithm.lower()
    if any(token in lower for token in CAT_DB):
        return "Databases"
    if any(token in lower for token in CAT_LIB):
        return "Libraries"
    return "Algorithms"


def pareto_frontier(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    # Collapse repeated measurements at the same Recall before building the
    # frontier.  This removes duplicate source rows without inventing data.
    best_at_recall: dict[float, float] = {}
    for recall, qps in points:
        best_at_recall[recall] = max(qps, best_at_recall.get(recall, -math.inf))
    points = sorted(best_at_recall.items(), key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, float]] = []
    best_qps = -math.inf
    for recall, qps in points:
        if qps > best_qps:
            selected.append((recall, qps))
            best_qps = qps
    selected.sort(key=lambda item: item[0])
    if not selected:
        return [], []
    recalls, qps_values = zip(*selected)
    return list(recalls), list(qps_values)


def y_setup(values: list[float]) -> tuple[float, float, list[float]]:
    values = [value for value in values if value > 0]
    if not values:
        return 1, 100, [1, 10, 100]
    minimum, maximum = min(values), max(values)
    top_exponent = math.ceil(math.log10(maximum * 1.1))
    top = 10**top_exponent
    if minimum < 1:
        bottom_exponent = -1
    else:
        bottom_exponent = 0
        if minimum > 100:
            bottom_exponent = math.floor(math.log10(minimum * 0.5))
    bottom = 10**bottom_exponent
    ticks = [10**exponent for exponent in range(bottom_exponent, top_exponent + 1)]
    if len(ticks) < 2:
        ticks = [bottom, top]
    return bottom, top, ticks


def x_setup(values: list[float]) -> tuple[float, float, list[float]]:
    if not values:
        return 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    lower = math.floor(min(values) * 10) / 10.0
    upper = math.ceil(max(values) * 10) / 10.0
    lower = max(0.0, lower)
    upper = min(1.0, upper) if upper <= 1.0 else upper
    if upper <= lower:
        upper = lower + 0.1
    return lower, upper, list(np.linspace(lower, upper, 5))


def format_recall(value: float, _position: int) -> str:
    if abs(value - 1.0) < 1e-9:
        return "1.0"
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.2g}"


def format_qps(value: float, _position: int) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def write_audit(
    frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    curves: dict[tuple[str, str, str], tuple[list[float], list[float]]],
) -> None:
    """Write raw, retained, and plotted-point counts for every series."""
    rows: list[dict] = []
    for dataset in DATASETS:
        for category in CATEGORIES:
            raw_subset = raw_frame[
                (raw_frame["dataset"] == dataset)
                & (raw_frame["category"] == category)
            ]
            kept_subset = frame[
                (frame["dataset"] == dataset)
                & (frame["category"] == category)
            ]
            algorithms = sorted(set(raw_subset["algorithm"]) | set(kept_subset["algorithm"]))
            for algorithm in algorithms:
                curve = curves.get((dataset, category, algorithm), ([], []))
                rows.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "algorithm": algorithm,
                        "raw_records": int((raw_subset["algorithm"] == algorithm).sum()),
                        "retained_records": int((kept_subset["algorithm"] == algorithm).sum()),
                        "pareto_points_plotted": len(curve[0]),
                    }
                )

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    summary = audit.groupby(["dataset", "category"])[
        ["raw_records", "retained_records", "pareto_points_plotted"]
    ].sum()
    print("[audit] raw / retained / plotted Pareto points:")
    print(summary.to_string())
    print(f"[audit] {AUDIT_CSV}")


def draw_figure(frame: pd.DataFrame, raw_frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    all_algorithms = sorted(frame["algorithm"].unique())
    algorithm_index = {algorithm: index for index, algorithm in enumerate(all_algorithms)}

    def style_for(algorithm: str) -> tuple[str, str]:
        lower = algorithm.lower()
        color = COLORS.get(algorithm, COLORS.get(lower, FALLBACK_COLORS[algorithm_index[algorithm] % len(FALLBACK_COLORS)]))
        marker = MARKERS.get(algorithm, MARKERS.get(lower, FALLBACK_MARKERS[algorithm_index[algorithm] % len(FALLBACK_MARKERS)]))
        return color, marker

    category_algorithms = {
        category: [algorithm for algorithm in all_algorithms if get_category(algorithm) == category]
        for category in CATEGORIES
    }

    # First collect the Pareto curves. Each panel gets its own QPS domain,
    # matching the original three-panel style and keeping sparse panels such
    # as msong-Databases readable.
    curves: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    for dataset in DATASETS:
        for category in CATEGORIES:
            for algorithm in category_algorithms[category]:
                subset = frame[
                    (frame["dataset"] == dataset)
                    & (frame["algorithm"] == algorithm)
                ]
                points = list(zip(subset["recall"].astype(float), subset["qps"].astype(float)))
                recalls, qps_values = pareto_frontier(points)
                if recalls:
                    curves[(dataset, category, algorithm)] = (recalls, qps_values)

    write_audit(frame, raw_frame, curves)

    fig, axes = plt.subplots(2, 3, figsize=(32, 19), squeeze=False)
    # Keep a compact shared legend above the panels and let the six axes use
    # the same Recall-QPS template without captions below each subplot.
    plt.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.76, wspace=0.24, hspace=0.34)
    for row, dataset in enumerate(DATASETS):
        for column, category in enumerate(CATEGORIES):
            ax = axes[row][column]
            recall_values: list[float] = []
            panel_qps_values: list[float] = []
            panel_has_data = False
            for algorithm in category_algorithms[category]:
                curve = curves.get((dataset, category, algorithm))
                if curve is None:
                    continue
                recalls, qps_values = curve
                color, marker = style_for(algorithm)
                # Draw a segment only when two actual frontier points exist;
                # then draw every frontier point explicitly so a line can
                # never appear to continue past an unmarked endpoint.
                if len(recalls) >= 2:
                    ax.plot(
                        recalls,
                        qps_values,
                        color=color,
                        linewidth=4.0,
                        alpha=0.82,
                        zorder=2,
                    )
                ax.scatter(
                    recalls,
                    qps_values,
                    color=color,
                    marker=marker,
                    s=115 if len(recalls) <= 40 else 62,
                    linewidths=1.4,
                    alpha=0.96,
                    zorder=3,
                )
                recall_values.extend(recalls)
                panel_qps_values.extend(qps_values)
                panel_has_data = True

            x_min, x_max, x_ticks = x_setup(recall_values)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(format_recall))

            y_min, y_max, y_ticks = y_setup(panel_qps_values)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(format_qps))
            ax.yaxis.set_minor_formatter(plt.NullFormatter())

            ax.set_xlabel("Recall", fontsize=40, fontweight="bold", labelpad=15)
            # Keep the shared y-axis title only on the leftmost panel of each row.
            # Tick values remain on every panel so the curves are still readable.
            if column == 0:
                ax.set_ylabel("QPS", fontsize=40, fontweight="bold", labelpad=15)
            else:
                ax.set_ylabel("")
            ax.set_title("")
            ax.set_box_aspect(0.75)
            ax.tick_params(axis="both", which="major", labelsize=32, width=3.0, length=10, pad=11)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_linewidth(2.8)

            if not panel_has_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=34,
                    fontweight="bold",
                )

    # Dataset labels make the requested row order explicit without repeating
    # the dataset name in every panel title.
    # Align row labels to the actual axes centers instead of hard-coding a
    # figure coordinate; this keeps cc_news/msong centered after layout edits.
    row_centers = [
        float((axes[row][0].get_position().y0 + axes[row][0].get_position().y1) / 2)
        for row in range(2)
    ]
    fig.text(0.042, row_centers[0], "W2", rotation=90, ha="center", va="center", fontsize=48, fontweight="bold")
    fig.text(0.042, row_centers[1], "S1", rotation=90, ha="center", va="center", fontsize=48, fontweight="bold")

    legend_algorithms = [
        algorithm
        for category in CATEGORIES
        for algorithm in category_algorithms[category]
        if (DATASETS[0], category, algorithm) in curves or (DATASETS[1], category, algorithm) in curves
    ]
    legend_handles = []
    for algorithm in legend_algorithms:
        color, marker = style_for(algorithm)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linewidth=5.5,
                markersize=15,
                label=DISPLAY_NAMES.get(algorithm, algorithm),
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.55, 0.98),
            ncol=7,
            frameon=False,
            prop={"size": 31, "weight": "bold"},
            columnspacing=1.0,
            handletextpad=0.45,
            labelspacing=0.65,
        )

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUTPUT_PNG}")
    print(f"[saved] {OUTPUT_PDF}")


if __name__ == "__main__":
    data, raw_data = load_records()
    draw_figure(data, raw_data)
