"""Generate the six main figures planned for Sections 5.6 and 5.7.

The script is intentionally fail-soft: a missing or semantically mismatched
source marks the relevant panel as skipped, records the reason, and continues
with the remaining figures.  All plotted points are exported alongside the
figures for auditability.
"""

from __future__ import annotations

import json
import math
import runpy
import shutil
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogFormatter, LogLocator, PercentFormatter


BASE_DIR = Path(__file__).resolve().parent
EXP_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR / "figures_5_6_5_7_20260918"

RESULTS_XLSX = EXP_DIR / "实验结果.xlsx"
DIAG_XLSX = EXP_DIR / "算法级细粒度分析.xlsx"
COST_XLSX = EXP_DIR / "w1_w4_s1_s8_total_parameter_cost_quality_20260916.xlsx"
SCALING_XLSX = EXP_DIR / "线程扩展.xlsx"
SIX_PANEL_SCRIPT = BASE_DIR / "cc_news_msong_六连图.py"
TWO_PANEL_SCRIPT = BASE_DIR / "二连图.py"

STATUS: list[dict[str, object]] = []

COLORS = {
    "ACORN": "#17becf",
    "DiGRA": "#756bb1",
    "iRangeGraph": "#d62728",
    "RangePQ": "#e377c2",
    "Faiss-HNSW": "#1f77b4",
    "HNSWlib-HNSW": "#ff7f0e",
    "Milvus-HNSW": "#2ca02c",
    "Weaviate-HNSW": "#8c564b",
    "Milvus-GPU-CAGRA": "#4c78a8",
    "GPU-CAGRA": "#4c78a8",
    "online UDF": "#1f77b4",
    "precomputed UDF": "#ff7f0e",
}

# A compact, reference-style palette for the two correlation figures.
# Keep the mapping stable across original/shuffled panels.
CORRELATION_COLORS = {
    "ACORN": "#1f77b4",
    "DiGRA": "#ff7f0e",
    "iRangeGraph": "#2ca02c",
}

CORRELATION_MARKERS = {
    "ACORN": "o",
    "DiGRA": "s",
    "iRangeGraph": "^",
}

# Query-only ACORN refresh completed on 186 with one search thread.  The
# retained indexes are the existing C1-C4 indexes; only ACORN's query points
# are replaced in the correlation pair figures.  DiGRA and iRangeGraph keep
# the formal data used by the original figures.
ACORN_SINGLE_THREAD_POINTS = {
    "artificial-corr-12-original": [
        (100, 0.8587, 193.1),
        (200, 0.9295, 112.2),
        (400, 0.9727, 66.3),
        (800, 0.9917, 39.4),
        (1600, 0.9980, 29.7),
    ],
    "artificial-corr-12-shuffled": [
        (100, 0.9657, 243.3),
        (200, 0.9888, 140.5),
        (400, 0.9970, 80.6),
        (800, 0.9993, 45.1),
        (1600, 0.9999, 27.8),
    ],
    "cc_news-original": [
        (100, 0.8649, 392.6),
        (200, 0.9010, 217.8),
        (400, 0.9169, 119.9),
        (800, 0.9238, 62.3),
        (1600, 0.9262, 31.0),
    ],
    "cc_news-shuffled": [
        (100, 0.8301, 351.2),
        (200, 0.8742, 195.5),
        (400, 0.8954, 105.3),
        (800, 0.9041, 54.9),
        (1600, 0.9073, 27.6),
    ],
}

RANGE_PAIR_COLORS = {
    "ACORN": "#1f77b4",
    "DiGRA": "#ff7f0e",
    "iRangeGraph": "#2ca02c",
    "RangePQ": "#9467bd",
    "Faiss-HNSW": "#d62728",
    "HNSWlib-HNSW": "#8c564b",
    "Milvus-HNSW": "#17becf",
}

THREAD_COLORS = {
    "ACORN": "#1f77b4",
    "HNSWlib-HNSW": "#ff7f0e",
    "Milvus-HNSW": "#2ca02c",
    "Weaviate-HNSW": "#9467bd",
}

THREAD_MARKERS = {
    "ACORN": "o",
    "HNSWlib-HNSW": "s",
    "Milvus-HNSW": "^",
    "Weaviate-HNSW": "D",
}

MARKERS = {
    "ACORN": "X",
    "DiGRA": "v",
    "iRangeGraph": "o",
    "RangePQ": "D",
    "Faiss-HNSW": "s",
    "HNSWlib-HNSW": "^",
    "Milvus-HNSW": "P",
    "Weaviate-HNSW": "o",
    "Milvus-GPU-CAGRA": "h",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.unicode_minus": False,
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "savefig.dpi": 600,
        }
    )


def add_status(
    figure_id: str,
    panel: str,
    status: str,
    source: str,
    note: str,
    n_rows: int | float | None = None,
) -> None:
    STATUS.append(
        {
            "figure": figure_id,
            "panel": panel,
            "status": status,
            "source": source,
            "n_rows": None if n_rows is None else int(n_rows),
            "note": note,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png, pdf


def save_data(frame: pd.DataFrame, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def style_axis(ax: plt.Axes, y_log: bool = False) -> None:
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="major", color="#b8b8b8", alpha=0.28, linewidth=0.8)
    ax.tick_params(axis="both", which="major", width=1.8, length=7, pad=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def caption(ax: plt.Axes, text: str, y: float = -0.28, size: int = 17) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=size,
        fontweight="bold",
        clip_on=False,
    )


def fmt_qps(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.2g}"


def compute_pareto_frontier(
    frame: pd.DataFrame,
    x_col: str = "recall",
    y_col: str = "qps",
) -> pd.DataFrame:
    """Return points that are not dominated when maximizing x and y."""
    grouped = frame.groupby(x_col, as_index=False)[y_col].max()
    ordered = grouped.sort_values([x_col, y_col], ascending=[False, False])
    best_y = -math.inf
    keep: list[bool] = []
    for row in ordered.itertuples(index=False):
        is_frontier = float(getattr(row, y_col)) > best_y
        keep.append(is_frontier)
        if is_frontier:
            best_y = float(getattr(row, y_col))
    return ordered.loc[keep].sort_values(x_col).reset_index(drop=True)


def select_build_quality_frontier(
    frame: pd.DataFrame,
    recall_tolerance: float = 1e-3,
) -> list[int]:
    """Select non-dominated build-time/Recall points per algorithm.

    Build time is minimized and Recall is maximized. The small Recall
    tolerance avoids retaining visually indistinguishable points caused by
    tiny measurement differences; it does not average or alter any value.
    """
    selected: list[int] = []
    for _algorithm, group in frame.groupby("algorithm", sort=False):
        ordered = group.sort_values(
            ["build_time_s", "recall", "qps"],
            ascending=[True, False, False],
        )
        best_recall = -math.inf
        for index, row in ordered.iterrows():
            recall = float(row["recall"])
            if recall > best_recall + recall_tolerance:
                selected.append(index)
                best_recall = recall
    return selected


def normalize_range_algorithm(value: object) -> str:
    key = str(value).strip().lower()
    mapping = {
        "acorn": "ACORN",
        "digra": "DiGRA",
        "irangegraph": "iRangeGraph",
        "rangepq": "RangePQ",
        "faiss-hnsw": "Faiss-HNSW",
        "faiss_hnsw": "Faiss-HNSW",
        "hnswlib": "HNSWlib-HNSW",
        "hnswlib-hnsw": "HNSWlib-HNSW",
        "milvus-hnsw": "Milvus-HNSW",
        "weaviate-hnsw": "Weaviate-HNSW",
        "milvus-gpu-cagra": "Milvus-GPU-CAGRA",
    }
    return mapping.get(key, str(value))


def run_existing_six_panel() -> None:
    figure_id = "5.6.1"
    try:
        if not SIX_PANEL_SCRIPT.exists():
            add_status(figure_id, "all", "skipped", str(SIX_PANEL_SCRIPT), "existing six-panel script not found")
            return
        runpy.run_path(str(SIX_PANEL_SCRIPT), run_name="__main__")
        copied = 0
        for suffix in ["png", "pdf", "csv"]:
            source = BASE_DIR / f"cc_news_msong_6panel.{suffix}" if suffix != "csv" else BASE_DIR / "cc_news_msong_6panel_audit.csv"
            if source.exists() and source.stat().st_size > 0:
                target_name = f"fig_5_6_1_cc_news_msong_6panel.{suffix}"
                shutil.copy2(source, OUT_DIR / target_name)
                copied += 1
        if copied < 2:
            add_status(figure_id, "all", "skipped", str(SIX_PANEL_SCRIPT), "script ran but no readable figure output was produced")
        else:
            add_status(
                figure_id,
                "all",
                "complete",
                str(SIX_PANEL_SCRIPT),
                "2x3 W2/S1 Recall-QPS Pareto figure regenerated with the shared template layout",
                copied,
            )
    except Exception as exc:  # keep the remaining figures running
        add_status(figure_id, "all", "failed", str(SIX_PANEL_SCRIPT), f"{type(exc).__name__}: {exc}")
        print(f"[skip {figure_id}] {traceback.format_exc()}")


def add_unavailable(ax: plt.Axes, panel_title: str, reason: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.60,
        panel_title,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        "Skipped",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#8a3b12",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.28,
        reason,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
        transform=ax.transAxes,
    )


def draw_workload_awareness() -> None:
    figure_id = "5.6.2"
    stem = "fig_5_6_2_workload_awareness"
    sp = pd.read_excel(COST_XLSX, sheet_name="Supplementary_search_points", header=3)
    sp["recall"] = pd.to_numeric(sp["recall"], errors="coerce")
    sp["qps"] = pd.to_numeric(sp["qps"], errors="coerce")
    sp["algorithm_label"] = sp["algorithm"].map(normalize_range_algorithm)
    sp = sp[sp["qps"].gt(0) & sp["recall"].between(0, 1.01)].copy()
    range_algorithms = [
        "ACORN",
        "DiGRA",
        "iRangeGraph",
        "RangePQ",
        "Faiss-HNSW",
        "HNSWlib-HNSW",
        "Milvus-HNSW",
        "Weaviate-HNSW",
    ]
    range_data = sp[
        sp["condition"].isin(["tight", "wide"])
        & sp["algorithm_label"].isin(range_algorithms)
    ].copy()

    formal = pd.read_excel(RESULTS_XLSX, sheet_name="Formal_Points")
    formal["algorithm_label"] = formal["algorithm"].map(normalize_range_algorithm)
    formal["recall"] = pd.to_numeric(formal["recall"], errors="coerce")
    formal["qps"] = pd.to_numeric(formal["qps"], errors="coerce")
    c_datasets = [
        "artificial-corr-12-original",
        "artificial-corr-12-shuffled",
        "cc_news-original",
        "cc_news-shuffled",
    ]
    d_data = formal[
        formal["dataset"].isin(c_datasets)
        & formal["algorithm_label"].isin(["ACORN", "DiGRA", "iRangeGraph"])
        & formal["recall"].ge(0.90)
        & formal["qps"].gt(0)
    ].copy()
    if not d_data.empty:
        d_data = (
            d_data.sort_values("qps")
            .groupby(["dataset", "algorithm_label"], as_index=False)
            .tail(1)
        )

    fig, axes = plt.subplots(2, 3, figsize=(23, 13), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.73, wspace=0.25, hspace=0.52)

    # (a) The local diagnostic workbook is MSong-Tight-12 only; it does not
    # contain the requested R1/S5 paired access counters.
    add_unavailable(axes[0, 0], "(a) Native filtering access", "no matched R1/S5 ACORN\ninvalid/valid access counters")
    add_status(
        figure_id,
        "a",
        "skipped",
        str(DIAG_XLSX),
        "available diagnostic is MSong-Tight-12 only; no matched R1/S5 ACORN access-ratio pair",
    )

    # (b) and (c) use the same x/y domain.  This keeps the range comparison
    # readable while retaining RangePQ's low-recall points.
    axes_bc = [axes[0, 1], axes[0, 2]]
    conditions = ["wide", "tight"]
    panel_names = ["(b) R1: Wide range", "(c) R2: Narrow range"]
    all_bc = range_data[range_data["condition"].isin(conditions)]
    if all_bc.empty:
        for ax, panel_name, condition in zip(axes_bc, panel_names, conditions):
            add_unavailable(ax, panel_name, "no readable tight/wide range points")
            add_status(figure_id, "b" if condition == "wide" else "c", "skipped", str(COST_XLSX), "no readable range points")
    else:
        qps_min = max(float(all_bc["qps"].min()) * 0.75, 0.1)
        qps_max = float(all_bc["qps"].max()) * 1.25
        plotted_algorithms: set[str] = set()
        for ax, panel_name, condition in zip(axes_bc, panel_names, conditions):
            panel = range_data[range_data["condition"] == condition]
            for algorithm in range_algorithms:
                group = panel[panel["algorithm_label"] == algorithm]
                if group.empty:
                    continue
                plotted_algorithms.add(algorithm)
                group = group.groupby("recall", as_index=False)["qps"].max().sort_values("recall")
                color = COLORS.get(algorithm, "#444444")
                marker = MARKERS.get(algorithm, "o")
                if len(group) >= 2:
                    ax.plot(group["recall"], group["qps"], color=color, linewidth=2.2, alpha=0.82, zorder=2)
                ax.scatter(
                    group["recall"],
                    group["qps"],
                    s=48,
                    color=color,
                    marker=marker,
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.95,
                    zorder=3,
                )
            ax.set_xlim(0.0, 1.02)
            ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2g}"))
            ax.set_ylim(qps_min, qps_max)
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
            ax.set_xlabel("Recall@100", fontweight="bold")
            if ax is axes[0, 1]:
                ax.set_ylabel("QPS", fontweight="bold")
            else:
                ax.set_ylabel("")
            style_axis(ax, y_log=True)
            caption(ax, panel_name)
            add_status(
                figure_id,
                "b" if condition == "wide" else "c",
                "complete",
                str(COST_XLSX),
                "tight/wide one-thread range-aware points; RangePQ low-recall points retained; coordinates shared",
                len(panel),
            )

    # (d) C1-C4 highest measured QPS at Recall >= 0.90.
    ax = axes[1, 0]
    if d_data.empty:
        add_unavailable(ax, "(d) Correlation and performance", "no C1-C4 point at Recall >= 0.90")
        add_status(figure_id, "d", "skipped", str(RESULTS_XLSX), "no valid C1-C4 points at the requested recall threshold")
    else:
        group_order = ["Artificial", "CC-News"]
        dataset_group = {
            "artificial-corr-12-original": "Artificial",
            "artificial-corr-12-shuffled": "Artificial",
            "cc_news-original": "CC-News",
            "cc_news-shuffled": "CC-News",
        }
        d_data["group"] = d_data["dataset"].map(dataset_group)
        d_data["variant"] = np.where(d_data["dataset"].str.endswith("original"), "original", "shuffled")
        method_offsets = {"ACORN": -0.12, "DiGRA": 0.0, "iRangeGraph": 0.12}
        variant_offsets = {"original": -0.035, "shuffled": 0.035}
        for algorithm in ["ACORN", "DiGRA", "iRangeGraph"]:
            for group_index, group_name in enumerate(group_order):
                segment = d_data[(d_data["algorithm_label"] == algorithm) & (d_data["group"] == group_name)]
                if segment.empty:
                    continue
                segment = segment.sort_values("variant")
                xs: list[float] = []
                ys: list[float] = []
                for _, row in segment.iterrows():
                    x = group_index + method_offsets[algorithm] + variant_offsets[row["variant"]]
                    xs.append(x)
                    ys.append(float(row["qps"]))
                    ax.scatter(
                        x,
                        row["qps"],
                        s=92,
                        color=COLORS[algorithm],
                        marker="o" if row["variant"] == "original" else "s",
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=4,
                    )
                if len(xs) == 2:
                    ax.plot(xs, ys, color=COLORS[algorithm], linewidth=1.5, linestyle=":", alpha=0.8, zorder=2)
        ax.set_xticks([0, 1], group_order)
        ax.set_ylabel("Highest measured QPS\n(Recall@100 ≥ 0.90)", fontweight="bold")
        ax.set_xlabel("Workload group", fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
        style_axis(ax, y_log=True)
        caption(ax, "(d) Correlation and performance")
        add_status(
            figure_id,
            "d",
            "complete",
            str(RESULTS_XLSX),
            "C1-C4 highest measured QPS at Recall@100 >= 0.90; original/shuffled paired within Artificial and CC-News",
            len(d_data),
        )

    # (e) and (f) require same-implementation online/precomputed matched runs.
    add_unavailable(axes[1, 1], "(e) Expensive predicate performance", "no same-implementation\nonline/precomputed pair")
    add_unavailable(axes[1, 2], "(f) Expensive predicate cost", "no matched predicate-cumulative\nand total-search time pair")
    for panel, reason in [
        ("e", "UDF workbook has online and precomputed paths across different implementations, not a same-implementation pair"),
        ("f", "UDF workbook lacks the requested matched predicate-cumulative and total-search time series"),
    ]:
        add_status(figure_id, panel, "skipped", str(RESULTS_XLSX), reason)

    legend_algorithms = [a for a in range_algorithms if a in set(range_data["algorithm_label"])]
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[a],
            marker=MARKERS.get(a, "o"),
            linewidth=2.5,
            markersize=8,
            label=a,
        )
        for a in legend_algorithms
    ]
    handles += [
        Line2D([0], [0], color="#555555", marker="o", linestyle="None", markersize=8, label="original"),
        Line2D([0], [0], color="#555555", marker="s", linestyle="None", markersize=8, label="shuffled"),
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.985),
            ncol=5,
            frameon=False,
            prop={"size": 13, "weight": "bold"},
            columnspacing=1.0,
            handletextpad=0.45,
        )
    save_figure(fig, stem)
    export = pd.concat(
        [
            range_data.assign(panel=np.where(range_data["condition"].eq("wide"), "b", "c")),
            d_data.assign(panel="d"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_data(export, "fig_5_6_2_workload_awareness_data")


def draw_range_pair() -> None:
    """Export the requested standalone R1/R2 Recall-QPS two-panel figure."""
    figure_id = "5.6.2-bc"
    stem = "fig_5_6_2_range_recall_qps_pair"
    sp = pd.read_excel(COST_XLSX, sheet_name="Supplementary_search_points", header=3)
    sp["recall"] = pd.to_numeric(sp["recall"], errors="coerce")
    sp["qps"] = pd.to_numeric(sp["qps"], errors="coerce")
    sp["algorithm_label"] = sp["algorithm"].map(normalize_range_algorithm)
    algorithms = [
        "iRangeGraph",
        "RangePQ",
        "Faiss-HNSW",
        "HNSWlib-HNSW",
        "Milvus-HNSW",
        "ACORN",
        "DiGRA",
    ]
    range_data = sp[
        sp["condition"].isin(["wide", "tight"])
        & sp["algorithm_label"].isin(algorithms)
        & sp["qps"].gt(0)
        & sp["recall"].between(0, 1.01)
    ].copy()
    if range_data.empty:
        add_status(figure_id, "all", "skipped", str(COST_XLSX), "no readable MSong-wide-80/MSong-tight-12 range points")
        return

    frontier_parts: list[pd.DataFrame] = []
    for condition in ["wide", "tight"]:
        for algorithm in algorithms:
            group = range_data[
                (range_data["condition"] == condition)
                & (range_data["algorithm_label"] == algorithm)
            ]
            if group.empty:
                continue
            frontier = compute_pareto_frontier(group)
            frontier["condition"] = condition
            frontier["algorithm_label"] = algorithm
            frontier_parts.append(frontier)
    frontier_data = pd.concat(frontier_parts, ignore_index=True)

    qps_min = max(float(frontier_data["qps"].min()) * 0.70, 0.1)
    qps_max = float(frontier_data["qps"].max()) * 1.35
    recall_min = max(
        0.0,
        math.floor((float(frontier_data["recall"].min()) - 0.02) * 20.0) / 20.0,
    )
    recall_max = min(
        1.02,
        math.ceil((float(frontier_data["recall"].max()) + 0.01) * 100.0) / 100.0,
    )
    if recall_min <= 0.0:
        recall_ticks = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    else:
        tick_start = math.ceil(recall_min * 20.0) / 20.0
        recall_ticks = np.arange(tick_start, recall_max + 0.001, 0.05)
        if len(recall_ticks) < 3:
            recall_ticks = np.linspace(recall_min, recall_max, 4)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.83, wspace=0.045)
    conditions = ["wide", "tight"]
    panel_titles = ["MSong-wide-80 (R1)", "MSong-tight-12 (R2)"]
    plotted: set[str] = set()
    for index, (ax, condition, panel_title) in enumerate(zip(axes, conditions, panel_titles)):
        panel = frontier_data[frontier_data["condition"] == condition]
        for algorithm in algorithms:
            group = panel[panel["algorithm_label"] == algorithm]
            if group.empty:
                continue
            plotted.add(algorithm)
            group = group.sort_values("recall")
            color = RANGE_PAIR_COLORS.get(algorithm, "#444444")
            marker = MARKERS.get(algorithm, "o")
            if len(group) >= 2:
                ax.plot(group["recall"], group["qps"], color=color, linewidth=2.8, alpha=0.90, zorder=2)
            ax.scatter(
                group["recall"],
                group["qps"],
                s=108,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.1,
                alpha=0.96,
                zorder=3,
            )
        ax.set_xlim(recall_min, recall_max)
        ax.set_xticks(recall_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
        ax.set_ylim(qps_min, qps_max)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
        ax.set_title(panel_title, fontsize=22, fontweight="bold", pad=10)
        ax.set_xlabel("Recall@100", fontsize=23, fontweight="bold")
        ax.set_ylabel("QPS" if index == 0 else "", fontsize=23, fontweight="bold")
        style_axis(ax, y_log=True)
        ax.set_box_aspect(0.75)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(18)
            tick.set_fontweight("bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=RANGE_PAIR_COLORS[a],
            marker=MARKERS.get(a, "o"),
            linewidth=2.8,
            markersize=12,
            label=a,
        )
        for a in algorithms
        if a in plotted
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        prop={"weight": "bold", "size": 15},
        columnspacing=0.85,
        handletextpad=0.35,
    )
    save_figure(fig, stem)
    save_data(frontier_data, f"{stem}_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(COST_XLSX),
        "standalone 1x2 Pareto-frontier figure for MSong-wide-80 (R1) and MSong-tight-12 (R2); shared Recall-QPS coordinates; RangePQ low-recall frontier points retained",
        len(frontier_data),
    )


def draw_s1_s5_algorithm_pair() -> None:
    """Redraw the S1/MSong and S5/artificial-0.8 algorithm comparison."""
    figure_id = "S1-S5-algorithms"
    stem = "fig_s1_s5_algorithm_recall_qps_pair"
    if not TWO_PANEL_SCRIPT.exists():
        add_status(figure_id, "all", "skipped", str(TWO_PANEL_SCRIPT), "legacy S1/S5 data parser not found")
        return

    try:
        parser_namespace = runpy.run_path(str(TWO_PANEL_SCRIPT), run_name="__s1_s5_data_loader__")
        raw = parser_namespace["parse_all_data"]()
    except Exception as exc:
        add_status(figure_id, "all", "failed", str(TWO_PANEL_SCRIPT), f"{type(exc).__name__}: {exc}")
        return

    if raw.empty:
        add_status(figure_id, "all", "skipped", str(TWO_PANEL_SCRIPT), "legacy parser returned no records")
        return

    def normalize_algorithm(value: object) -> str | None:
        key = str(value).strip().lower()
        if key == "acorn":
            return "ACORN"
        if "digra" in key:
            return "DiGRA"
        if "irangegraph" in key:
            return "iRangeGraph"
        if "rangepq" in key:
            return "RangePQ"
        return None

    raw["recall"] = pd.to_numeric(raw["recall"], errors="coerce")
    raw["qps"] = pd.to_numeric(raw["qps"], errors="coerce")
    raw["algorithm_label"] = raw["algorithm"].map(normalize_algorithm)
    data = raw[
        raw["dataset"].isin(["msong", "artificial-0.8"])
        & raw["algorithm_label"].notna()
        & raw["recall"].between(0, 1.01)
        & raw["qps"].gt(0)
    ].copy()
    if data.empty:
        add_status(figure_id, "all", "skipped", str(TWO_PANEL_SCRIPT), "no valid S1/S5 algorithm Recall-QPS points")
        return

    algorithms = ["ACORN", "DiGRA", "iRangeGraph", "RangePQ"]
    panel_names = [("msong", "S1: MSong"), ("artificial-0.8", "S5: Artificial-0.8")]
    marker_map = {"ACORN": "o", "DiGRA": "s", "iRangeGraph": "^", "RangePQ": "D"}
    frontier_parts: list[pd.DataFrame] = []
    for dataset, _title in panel_names:
        for algorithm in algorithms:
            group = data[
                (data["dataset"] == dataset)
                & (data["algorithm_label"] == algorithm)
            ]
            if group.empty:
                continue
            frontier = compute_pareto_frontier(group)
            frontier["dataset"] = dataset
            frontier["algorithm_label"] = algorithm
            frontier_parts.append(frontier)
    if not frontier_parts:
        add_status(figure_id, "all", "skipped", str(TWO_PANEL_SCRIPT), "no S1/S5 Pareto-frontier points")
        return
    frontier_data = pd.concat(frontier_parts, ignore_index=True)

    qps_min = max(float(frontier_data["qps"].min()) * 0.70, 1.0)
    qps_max = float(frontier_data["qps"].max()) * 1.35
    recall_min = max(
        0.0,
        math.floor((float(frontier_data["recall"].min()) - 0.02) * 20.0) / 20.0,
    )
    recall_max = min(
        1.02,
        math.ceil((float(frontier_data["recall"].max()) + 0.01) * 100.0) / 100.0,
    )
    if recall_min <= 0.0:
        recall_ticks = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
    else:
        tick_start = math.ceil(recall_min * 20.0) / 20.0
        recall_ticks = np.arange(tick_start, recall_max + 0.001, 0.05)
        if len(recall_ticks) < 3:
            recall_ticks = np.linspace(recall_min, recall_max, 4)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.83, wspace=0.045)
    plotted: set[str] = set()
    for index, (ax, (dataset, panel_title)) in enumerate(zip(axes, panel_names)):
        panel = frontier_data[frontier_data["dataset"] == dataset]
        for algorithm in algorithms:
            group = panel[panel["algorithm_label"] == algorithm].sort_values("recall")
            if group.empty:
                continue
            plotted.add(algorithm)
            ax.plot(
                group["recall"],
                group["qps"],
                color=RANGE_PAIR_COLORS[algorithm],
                linewidth=2.8,
                alpha=0.90,
                zorder=2,
            )
            ax.scatter(
                group["recall"],
                group["qps"],
                s=112,
                color=RANGE_PAIR_COLORS[algorithm],
                marker=marker_map[algorithm],
                edgecolor="white",
                linewidth=1.1,
                alpha=0.96,
                zorder=3,
            )
        ax.set_xlim(recall_min, recall_max)
        ax.set_xticks(recall_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
        ax.set_ylim(qps_min, qps_max)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
        ax.set_title(panel_title, fontsize=22, fontweight="bold", pad=10)
        ax.set_xlabel("Recall@100", fontsize=23, fontweight="bold")
        ax.set_ylabel("QPS" if index == 0 else "", fontsize=23, fontweight="bold")
        style_axis(ax, y_log=False)
        ax.set_box_aspect(0.75)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(18)
            tick.set_fontweight("bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=RANGE_PAIR_COLORS[algorithm],
            marker=marker_map[algorithm],
            linewidth=2.8,
            markersize=13,
            label=algorithm,
        )
        for algorithm in algorithms
        if algorithm in plotted
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        prop={"weight": "bold", "size": 17},
        columnspacing=1.25,
        handletextpad=0.4,
    )
    save_figure(fig, stem)
    save_data(frontier_data, f"{stem}_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(TWO_PANEL_SCRIPT),
        "S1/MSong and S5/artificial-0.8 algorithm Recall-QPS Pareto-frontier pair; dominated points omitted",
        len(frontier_data),
    )


def draw_correlation_pair(
    figure_id: str,
    stem: str,
    display_name: str,
    dataset_pair: tuple[str, str],
) -> None:
    """Draw original/shuffled paired panels for one correlation workload."""
    formal = pd.read_excel(RESULTS_XLSX, sheet_name="Formal_Points")
    formal["algorithm_label"] = formal["algorithm"].map(normalize_range_algorithm)
    formal["recall"] = pd.to_numeric(formal["recall"], errors="coerce")
    formal["qps"] = pd.to_numeric(formal["qps"], errors="coerce")
    methods = ["ACORN", "DiGRA", "iRangeGraph"]
    selected = formal[
        formal["dataset"].isin(dataset_pair)
        & formal["algorithm_label"].isin(methods)
        & formal["recall"].between(0, 1.01)
        & formal["qps"].gt(0)
    ].copy()
    # Refresh only ACORN with the validated one-thread query-only rerun.  This
    # keeps the paired figures internally consistent while leaving the other
    # two methods on their existing formal-data frontier.
    acorn_rows = []
    for dataset in dataset_pair:
        for ef_search, recall, qps in ACORN_SINGLE_THREAD_POINTS.get(dataset, []):
            acorn_rows.append(
                {
                    "dataset": dataset,
                    "algorithm_label": "ACORN",
                    "recall": recall,
                    "qps": qps,
                    "efSearch": ef_search,
                    "threads": 1,
                    "source_note": "ACORN single-thread query-only reuse run on server 186",
                }
            )
    if acorn_rows:
        selected = selected[selected["algorithm_label"].ne("ACORN")].copy()
        selected = pd.concat([selected, pd.DataFrame(acorn_rows)], ignore_index=True, sort=False)
    if selected.empty:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no valid original/shuffled Recall-QPS points")
        return
    variant_map = {dataset_pair[0]: "original", dataset_pair[1]: "shuffled"}
    frontier_parts: list[pd.DataFrame] = []
    for dataset in dataset_pair:
        for method in methods:
            group = selected[
                (selected["dataset"] == dataset)
                & (selected["algorithm_label"] == method)
            ]
            if group.empty:
                continue
            frontier = compute_pareto_frontier(group)
            frontier["dataset"] = dataset
            frontier["algorithm_label"] = method
            frontier["variant"] = variant_map[dataset]
            frontier_parts.append(frontier)
    if not frontier_parts:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no Pareto-frontier points")
        return
    frontier_data = pd.concat(frontier_parts, ignore_index=True)

    qps_min = max(float(frontier_data["qps"].min()) * 0.70, 1.0)
    qps_max = float(frontier_data["qps"].max()) * 1.35
    recall_min = max(
        0.0,
        math.floor((float(frontier_data["recall"].min()) - 0.02) * 20.0) / 20.0,
    )
    recall_max = min(
        1.02,
        math.ceil((float(frontier_data["recall"].max()) + 0.01) * 100.0) / 100.0,
    )
    tick_start = math.ceil(recall_min * 20.0) / 20.0
    recall_ticks = np.arange(tick_start, recall_max + 0.001, 0.05)
    if len(recall_ticks) < 3:
        recall_ticks = np.linspace(recall_min, recall_max, 4)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.17, top=0.83, wspace=0.045)
    panel_titles = [f"{display_name} original", f"{display_name} shuffled"]
    for index, (ax, variant, panel_title) in enumerate(zip(axes, ["original", "shuffled"], panel_titles)):
        for method in methods:
            group = frontier_data[
                (frontier_data["variant"] == variant)
                & (frontier_data["algorithm_label"] == method)
            ].sort_values("recall")
            if group.empty:
                continue
            ax.plot(
                group["recall"],
                group["qps"],
                color=CORRELATION_COLORS[method],
                linewidth=2.8,
                alpha=0.90,
                zorder=2,
            )
            ax.scatter(
                group["recall"],
                group["qps"],
                s=112,
                color=CORRELATION_COLORS[method],
                marker=CORRELATION_MARKERS.get(method, "o"),
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
        ax.set_xlim(recall_min, recall_max)
        ax.set_xticks(recall_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.2f}"))
        ax.set_ylim(qps_min, qps_max)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
        ax.set_title(panel_title, fontsize=22, fontweight="bold", pad=10)
        ax.set_xlabel("Recall@100", fontsize=23, fontweight="bold")
        ax.set_ylabel("QPS" if index == 0 else "", fontsize=23, fontweight="bold")
        style_axis(ax, y_log=False)
        ax.set_box_aspect(0.75)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(18)
            tick.set_fontweight("bold")

    handles = [
        Line2D(
            [0],
            [0],
            color=CORRELATION_COLORS[method],
            marker=CORRELATION_MARKERS.get(method, "o"),
            linewidth=2.6,
            markersize=13,
            label=method,
        )
        for method in methods
        if method in set(frontier_data["algorithm_label"])
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=len(handles),
        frameon=False,
        prop={"weight": "bold", "size": 17},
        columnspacing=1.25,
        handletextpad=0.4,
    )
    save_figure(fig, stem)
    save_data(frontier_data, f"{stem}_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(RESULTS_XLSX),
        f"{display_name} original/shuffled paired Recall-QPS Pareto-frontier panels; dominated points omitted",
        len(frontier_data),
    )


def draw_cc_news_correlation_pair() -> None:
    draw_correlation_pair(
        "5.6.2-d-cc-news",
        "fig_5_6_2_correlation_cc_news_pair",
        "CC-News",
        ("cc_news-original", "cc_news-shuffled"),
    )


def draw_artificial_correlation_pair() -> None:
    draw_correlation_pair(
        "5.6.2-d-artificial",
        "fig_5_6_2_correlation_artificial_pair",
        "Artificial-corr-12",
        ("artificial-corr-12-original", "artificial-corr-12-shuffled"),
    )


def draw_expensive_predicate_performance() -> None:
    """Draw the available U1 Recall-QPS diagnostic with explicit caveats."""
    figure_id = "5.6.2-e-standalone"
    stem = "fig_5_6_2_expensive_predicate_performance"
    bvb = pd.read_excel(RESULTS_XLSX, sheet_name="UDF_BVB_Points")
    bvb = bvb[bvb["dataset"].eq("artificial-udf-12")].copy()
    bvb["ef"] = pd.to_numeric(
        bvb["name"].astype(str).str.extract(r"(?:efSearch|search_ef)[=:]([0-9]+)")[0],
        errors="coerce",
    )
    bvb["qps"] = pd.to_numeric(bvb["qps"], errors="coerce")
    bvb["recall"] = pd.to_numeric(bvb["recall_mean"], errors="coerce")
    specs = [
        ("elasticsearch-hnsw", "Elasticsearch-HNSW (online)", "nlinks:128, efConstruction:100", "online"),
        ("hnswlib-hnsw", "HNSWlib-HNSW (online)", "M:128 efConstruction:100", "online"),
        ("milvus-hnsw", "Milvus-HNSW (precomputed)", "index_M:128, index_ef:100", "precomputed"),
    ]
    parts: list[pd.DataFrame] = []
    for algorithm, label, pattern, semantic in specs:
        part = bvb[
            bvb["algorithm"].eq(algorithm)
            & bvb["name"].astype(str).str.contains(pattern, regex=False, na=False)
            & bvb["ef"].notna()
            & bvb["qps"].gt(0)
        ].copy()
        if part.empty:
            continue
        part["algorithm_label"] = label
        part["semantic"] = semantic
        parts.append(part[["algorithm_label", "semantic", "ef", "qps", "recall"]])

    acorn = pd.read_excel(RESULTS_XLSX, sheet_name="UDF_ACORN")
    acorn = acorn[
        acorn["dataset"].eq("artificial-udf-12")
        & acorn["M"].eq(128)
        & acorn["M_beta"].eq(256)
        & acorn["efConstruction"].eq(100)
        & acorn["gamma"].eq(20)
    ].copy()
    acorn["ef"] = pd.to_numeric(acorn["efSearch"], errors="coerce")
    acorn["qps"] = pd.to_numeric(acorn["qps"], errors="coerce")
    acorn["recall"] = pd.to_numeric(acorn["recall_at_100"], errors="coerce")
    if not acorn.empty:
        acorn["algorithm_label"] = "ACORN (precomputed)"
        acorn["semantic"] = "precomputed"
        parts.append(acorn[["algorithm_label", "semantic", "ef", "qps", "recall"]])

    if not parts:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no readable U1 Recall-QPS data")
        return
    data = pd.concat(parts, ignore_index=True)
    data = data[data["ef"].isin([100, 200, 400, 800, 1600])].copy()
    data = data.sort_values(["algorithm_label", "ef"])
    # The available U1 points are concentrated in the high-recall region.
    # Keep the view focused, while leaving a small right margin after Recall=1.
    recall_min = 0.94
    recall_max = 1.002

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.84)
    order = [
        "Elasticsearch-HNSW (online)",
        "HNSWlib-HNSW (online)",
        "Milvus-HNSW (precomputed)",
        "ACORN (precomputed)",
    ]
    colors = {
        "Elasticsearch-HNSW (online)": "#1f77b4",
        "HNSWlib-HNSW (online)": "#ff7f0e",
        "Milvus-HNSW (precomputed)": "#2ca02c",
        "ACORN (precomputed)": COLORS["ACORN"],
    }
    markers = {
        "Elasticsearch-HNSW (online)": "s",
        "HNSWlib-HNSW (online)": "^",
        "Milvus-HNSW (precomputed)": "P",
        "ACORN (precomputed)": "X",
    }
    for label in order:
        part = data[data["algorithm_label"] == label].sort_values("recall")
        if part.empty:
            continue
        ax.plot(
            part["recall"],
            part["qps"],
            color=colors[label],
            marker=markers[label],
            markersize=12,
            linewidth=2.8,
            label=label,
        )
    ax.set_xlim(recall_min, recall_max)
    ax.set_xticks([0.94, 0.96, 0.98, 1.00])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.2f}"))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
    ax.set_xlabel("Recall@100", fontsize=23, fontweight="bold")
    ax.set_ylabel("QPS", fontsize=23, fontweight="bold")
    style_axis(ax, y_log=False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight("bold")
    ax.set_box_aspect(0.75)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        prop={"weight": "bold", "size": 16},
        columnspacing=1.4,
        handletextpad=0.45,
    )
    save_figure(fig, stem)
    save_data(data, f"{stem}_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(RESULTS_XLSX),
        "available U1 Recall-QPS diagnostic; fixed representative build configurations; online/precomputed implementations are not same-implementation pairs; one run per point",
        len(data),
    )


def rename_scaling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "algorithm",
        "config_id",
        "build_parameters",
        "search_parameters",
        "selection",
        "T",
        "final_recall",
        "tuning_qps",
        "tuning_recall",
        "qps_mean",
        "qps_sd",
        "cv",
        "p50_us",
        "p95_us",
        "p99_us",
        "wall_time_s",
        "n",
        "data_source",
        "notes",
    ]
    if len(frame.columns) != len(columns):
        raise ValueError(f"unexpected Scaling_Detail columns: {len(frame.columns)}")
    frame = frame.copy()
    frame.columns = columns
    return frame


def rms_sd(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def draw_thread_scaling() -> None:
    figure_id = "5.6.3"
    data = rename_scaling_columns(pd.read_excel(SCALING_XLSX, sheet_name="Scaling_Detail", header=3))
    data["T"] = pd.to_numeric(data["T"], errors="coerce")
    for col in ["final_recall", "qps_mean", "qps_sd"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    labels = ["ACORN", "Milvus-HNSW", "Weaviate-HNSW"]
    selected_configs = {
        "ACORN": "ACORN_gamma100",
        "Milvus-HNSW": "Milvus_M64_efC100_ef1600",
        "Weaviate-HNSW": "Weaviate_M32_efC100",
    }
    thread_values = [1, 7, 14, 28]
    data = data[
        data["algorithm"].isin(labels)
        & data["T"].isin(thread_values)
        & data.apply(lambda row: row["config_id"] == selected_configs.get(row["algorithm"]), axis=1)
    ].copy()
    if data.empty:
        add_status(figure_id, "all", "skipped", str(SCALING_XLSX), "no readable thread-scaling rows")
        return
    agg = (
        data.groupby(["algorithm", "T"], as_index=False)
        .agg(
            qps=("qps_mean", "mean"),
            qps_sd=("qps_sd", rms_sd),
            recall=("final_recall", "mean"),
            selected_points=("config_id", "nunique"),
        )
        .sort_values(["algorithm", "T"])
    )
    x_positions = np.arange(len(thread_values), dtype=float)
    x_position_map = dict(zip(thread_values, x_positions))
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.83)
    for algorithm in labels:
        g = agg[agg["algorithm"] == algorithm].sort_values("T")
        if g.empty:
            continue
        x = g["T"].map(x_position_map).to_numpy(dtype=float)
        color = THREAD_COLORS[algorithm]
        ax.errorbar(
            x,
            g["qps"],
            yerr=g["qps_sd"],
            color=color,
            marker=THREAD_MARKERS[algorithm],
            linewidth=2.8,
            markersize=11,
            capsize=5,
            capthick=1.7,
            elinewidth=1.7,
            label=algorithm,
        )
    ax.set_title("Absolute throughput vs. concurrent workers", fontsize=22, fontweight="bold", pad=10)
    ax.set_ylabel("QPS", fontsize=23, fontweight="bold")
    ax.set_xlabel("Concurrent query workers (T)", fontsize=23, fontweight="bold")
    ax.set_xticks(x_positions, [str(value) for value in thread_values])
    ax.set_xlim(-0.20, len(thread_values) - 0.80)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
    ax.set_yscale("log")
    style_axis(ax, y_log=False)
    ax.set_box_aspect(0.75)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight("bold")
    handles = [
        Line2D(
            [0],
            [0],
            color=THREAD_COLORS[algorithm],
            marker=THREAD_MARKERS[algorithm],
            linewidth=2.8,
            markersize=13,
            label=algorithm,
        )
        for algorithm in labels
        if not agg[agg["algorithm"] == algorithm].empty
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        prop={"size": 17, "weight": "bold"},
        columnspacing=1.25,
        handletextpad=0.4,
    )
    save_figure(fig, "fig_5_6_3_thread_scaling")
    save_data(agg, "fig_5_6_3_thread_scaling_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(SCALING_XLSX),
        "App-Reviews absolute QPS for fixed representative configurations; HNSWlib-HNSW omitted; T={1,7,14,28} shown at equal x-spacing; error bars use run SD",
        len(agg),
    )


def rename_lowcost_runs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["run"] = pd.to_numeric(frame["run"], errors="coerce")
    frame["qps"] = pd.to_numeric(frame["qps"], errors="coerce")
    return frame


def draw_stability() -> None:
    figure_id = "5.6.4"
    runs = rename_lowcost_runs(pd.read_excel(RESULTS_XLSX, sheet_name="LowCost_Runs"))
    datasets = [
        "artificial-corr-12-original",
        "artificial-corr-12-shuffled",
        "cc_news-original",
        "cc_news-shuffled",
    ]
    runs = runs[(runs["kind"] == "query") & runs["dataset"].isin(datasets) & runs["qps"].gt(0)].copy()
    if runs.empty:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no C1-C4 repeated query runs")
        return
    config_stats = (
        runs.groupby(["dataset", "algorithm", "result_file"], as_index=False)
        .agg(n=("qps", "count"), qps_mean=("qps", "mean"), qps_sd=("qps", "std"))
    )
    config_stats["cv"] = config_stats["qps_sd"] / config_stats["qps_mean"]
    config_stats["dataset_label"] = config_stats["dataset"].map(
        {
            "artificial-corr-12-original": "C1\nArtificial\noriginal",
            "artificial-corr-12-shuffled": "C2\nArtificial\nshuffled",
            "cc_news-original": "C3\nCC-News\noriginal",
            "cc_news-shuffled": "C4\nCC-News\nshuffled",
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.18, top=0.84)
    box_groups = [config_stats[config_stats["dataset"] == d]["cv"].dropna().values for d in datasets]
    positions = np.arange(1, 5)
    box_colors = ["#5aa6b8", "#9fc9be", "#e5b56a", "#b7a5cf"]
    edge_colors = ["#2b6576", "#5f8177", "#9e7132", "#74628f"]
    box = ax.boxplot(
        box_groups,
        positions=positions,
        widths=0.42,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#24343a", "linewidth": 2.3},
        whiskerprops={"color": "#53666b", "linewidth": 1.8},
        capprops={"color": "#53666b", "linewidth": 1.8},
        boxprops={"linewidth": 1.7},
    )
    for patch, face, edge in zip(box["boxes"], box_colors, edge_colors):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
        patch.set_alpha(0.88)
    rng = np.random.default_rng(20260918)
    for position, values in zip(positions, box_groups):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(
            values,
            position + jitter,
            s=34,
            color="#30383b",
            alpha=0.56,
            zorder=4,
            linewidth=0.25,
        )
    ax.axvline(0.10, color="#b22222", linestyle="--", linewidth=2.2, zorder=1)
    ax.text(
        0.105,
        0.97,
        "10% reference",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        color="#b22222",
        fontsize=16,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_yticks(
        positions,
        ["C1  Artificial original", "C2  Artificial shuffled", "C3  CC-News original", "C4  CC-News shuffled"],
    )
    ax.invert_yaxis()
    ax.set_xlabel("QPS coefficient of variation", fontsize=23, fontweight="bold")
    ax.set_ylabel("Correlation workload", fontsize=23, fontweight="bold")
    ax.set_title("QPS variation across correlation workloads", fontsize=22, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    max_cv = float(config_stats["cv"].max()) if config_stats["cv"].notna().any() else 0.2
    ax.set_xlim(0, max(0.20, max_cv * 1.18))
    ax.set_ylim(4.55, 0.45)
    style_axis(ax)
    ax.set_box_aspect(0.58)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(17)
        tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(18)
        tick.set_fontweight("bold")
    save_figure(fig, "fig_5_6_4_stability")
    save_data(config_stats, "fig_5_6_4_cv_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(RESULTS_XLSX),
        "C1-C4 QPS coefficient-of-variation distribution panel only; 26 repeated configurations per workload",
        len(config_stats),
    )


def draw_build_cost_decomposition() -> None:
    figure_id = "5.7-a"
    build = pd.read_excel(RESULTS_XLSX, sheet_name="LowCost_Build")
    algorithms = [
        "hnswlib-hnsw",
        "faiss-hnsw",
        "milvus-hnsw",
        "weaviate-hnsw",
        "milvus-gpu-cagra",
        "milvus-gpu-ivfpq",
    ]
    build = build[
        (build["dataset"] == "cc_news-shuffled")
        & build["algorithm"].isin(algorithms)
    ].copy()
    for col in ["data_load_s", "index_build_s", "total_build_s", "build_threads"]:
        build[col] = pd.to_numeric(build[col], errors="coerce")
    build = build[build["build_threads"].eq(28)].copy()
    if build.empty or build[["data_load_s", "index_build_s"]].dropna(how="all").empty:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no complete C4 28-thread build-stage records")
        return
    labels = {
        "hnswlib-hnsw": "HNSWlib-HNSW",
        "faiss-hnsw": "Faiss-HNSW",
        "milvus-hnsw": "Milvus-HNSW",
        "weaviate-hnsw": "Weaviate-HNSW",
        "milvus-gpu-cagra": "GPU-CAGRA",
        "milvus-gpu-ivfpq": "GPU-IVFPQ",
    }
    summary = build.groupby("algorithm", as_index=False).agg(
        data_load_s=("data_load_s", "mean"),
        index_build_s=("index_build_s", "mean"),
        data_load_sd=("data_load_s", "std"),
        index_build_sd=("index_build_s", "std"),
        total_build_s=("total_build_s", "mean"),
        total_build_sd=("total_build_s", "std"),
        n=("build_repeat", "count"),
    )
    summary["algorithm_label"] = summary["algorithm"].map(labels)
    summary["total_build_s"] = summary["total_build_s"].fillna(summary["data_load_s"] + summary["index_build_s"])
    summary["total_build_sd"] = summary["total_build_sd"].fillna(0.0)
    summary = summary.set_index("algorithm").reindex(algorithms).reset_index()
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.78)
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["data_load_s"],
        width=0.58,
        color="#7db7d4",
        edgecolor="#24566e",
        linewidth=1.3,
        label="Data loading",
    )
    ax.bar(
        x,
        summary["index_build_s"],
        bottom=summary["data_load_s"],
        width=0.58,
        color="#e9a15b",
        edgecolor="#874a12",
        linewidth=1.3,
        label="Explicit index construction",
    )
    for xi, row in zip(x, summary.itertuples()):
        total = float(row.data_load_s + row.index_build_s)
        ax.errorbar(
            xi,
            total,
            yerr=float(row.total_build_sd) if np.isfinite(row.total_build_sd) else 0.0,
            fmt="none",
            ecolor="#27343a",
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=5,
        )
        ax.text(xi, total * 1.035, f"n={int(row.n)}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_xticks(x, summary["algorithm_label"])
    ax.set_ylabel("Build time (s)", fontsize=21, fontweight="bold")
    ax.set_xlabel("C4: cc_news-shuffled; 28 build threads", fontsize=19, fontweight="bold")
    ax.set_ylim(0, float((summary["data_load_s"] + summary["index_build_s"]).max()) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_qps))
    style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        prop={"size": 15, "weight": "bold"},
        columnspacing=1.4,
        handletextpad=0.45,
    )
    save_figure(fig, "fig_5_7_a_build_cost_decomposition")
    save_data(summary, "fig_5_7_a_build_cost_decomposition_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(RESULTS_XLSX),
        "186/28-thread C4 cc_news-shuffled; six algorithms; 3 independent build repeats; data load plus explicit index construction",
        len(build),
    )


def draw_build_quality() -> None:
    figure_id = "5.7-b"
    formal = pd.read_excel(RESULTS_XLSX, sheet_name="Formal_Points")
    algorithms = [
        "acorn",
        "digra",
        "irangegraph",
        "rangepq",
        "hnswlib-hnsw",
        "faiss-hnsw",
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
    ]
    labels = {
        "acorn": "ACORN",
        "digra": "DiGRA",
        "irangegraph": "iRangeGraph",
        "rangepq": "RangePQ",
        "hnswlib-hnsw": "HNSWlib-HNSW",
        "faiss-hnsw": "Faiss-HNSW",
        "milvus-hnsw": "Milvus-HNSW",
        "milvus-ivfpq": "Milvus-IVFPQ",
        "milvus-gpu-ivfpq": "GPU-IVFPQ",
        "milvus-gpu-cagra": "GPU-CAGRA",
        "weaviate-hnsw": "Weaviate-HNSW",
        "qdrant": "Qdrant",
        "vearch-ivfpq": "Vearch-IVFPQ",
        "vearch-hnsw": "Vearch-HNSW",
        "redis-hnsw": "Redis-HNSW",
        "elasticsearch-hnsw": "Elasticsearch-HNSW",
        "pgvector-hnsw": "PGVector-HNSW",
    }
    markers = {
        "acorn": "o",
        "digra": "s",
        "irangegraph": "^",
        "rangepq": "D",
        "hnswlib-hnsw": "+",
        "faiss-hnsw": "P",
        "milvus-hnsw": "X",
        "milvus-ivfpq": "v",
        "milvus-gpu-ivfpq": "<",
        "milvus-gpu-cagra": ">",
        "weaviate-hnsw": "h",
        "qdrant": "H",
        "vearch-ivfpq": "8",
        "vearch-hnsw": "p",
        "redis-hnsw": "*",
        "elasticsearch-hnsw": "d",
        "pgvector-hnsw": ".",
    }
    plot_data = formal[
        (formal["dataset"] == "artificial-corr-12-original")
        & formal["algorithm"].isin(algorithms)
    ].copy()
    for col in ["build_time_s", "qps", "recall"]:
        plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")
    if "valid" in plot_data.columns:
        valid = plot_data["valid"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
        plot_data = plot_data[valid].copy()
    plot_data = plot_data[
        plot_data["build_time_s"].gt(0)
        & plot_data["recall"].notna()
        & plot_data["qps"].gt(0)
    ].copy()
    group_cols = ["algorithm", "build_parameters_json"]
    merged_counts = (
        plot_data.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "merged_query_points"})
    )
    plot_data = (
        plot_data.sort_values(
            group_cols + ["recall", "qps"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(group_cols, keep="first")
        .merge(merged_counts, on=group_cols, how="left")
    )
    if plot_data.empty:
        add_status(figure_id, "all", "skipped", str(RESULTS_XLSX), "no complete C1 formal build-time/Recall records")
        return
    all_merged_data = plot_data.copy()
    frontier_indices = select_build_quality_frontier(all_merged_data)
    all_merged_data["pareto_frontier"] = all_merged_data.index.isin(frontier_indices)
    plot_data = all_merged_data[all_merged_data["pareto_frontier"]].copy()
    fig, ax = plt.subplots(figsize=(15.2, 8.8))
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.17, top=0.84)
    vmin = float(plot_data["qps"].min())
    vmax = float(plot_data["qps"].max())
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for algorithm in algorithms:
        panel = plot_data[plot_data["algorithm"] == algorithm]
        if panel.empty:
            continue
        for _, row in panel.iterrows():
            color = cmap(norm(row["qps"]))
            scatter_kwargs = {
                "s": 54,
                "marker": markers[algorithm],
                "color": color,
                "alpha": 0.68,
                "zorder": 3,
            }
            if markers[algorithm] not in {"+", "."}:
                scatter_kwargs.update(edgecolor="white", linewidth=0.45)
            ax.scatter(row["build_time_s"], row["recall"], **scatter_kwargs)
    ax.set_xlabel("Build time (s, log scale)", fontsize=21, fontweight="bold")
    ax.set_ylabel("Recall@100", fontsize=21, fontweight="bold")
    x_min = float(plot_data["build_time_s"].min())
    x_max = float(plot_data["build_time_s"].max())
    ax.set_xscale("log")
    ax.set_xlim(max(0.1, x_min * 0.85), x_max * 1.15)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.xaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
    y_min = float(plot_data["recall"].min())
    ax.set_ylim(max(0.0, y_min - 0.04), 1.04)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    style_axis(ax)
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[algorithm],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            markersize=8,
            label=labels[algorithm],
        )
        for algorithm in algorithms
        if not plot_data[plot_data["algorithm"] == algorithm].empty
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.98),
        ncol=5,
        frameon=False,
        prop={"size": 12.5, "weight": "bold"},
        columnspacing=1.0,
        handletextpad=0.45,
        labelspacing=0.65,
    )
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("QPS", fontsize=18, fontweight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=14, width=1.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(17)
        tick.set_fontweight("bold")
    save_figure(fig, "fig_5_7_b_build_time_recall")
    save_data(plot_data, "fig_5_7_b_build_time_recall_data")
    save_data(all_merged_data, "fig_5_7_b_build_time_recall_all_merged_data")
    add_status(
        figure_id,
        "all",
        "complete",
        str(RESULTS_XLSX),
        f"186/28-thread C1 artificial-corr-12-original formal grid; all 17 algorithms; 444 raw points merged to {len(all_merged_data)} build-parameter points, then {len(plot_data)} per-algorithm build-time/Recall Pareto points; no averaging; color encodes QPS",
        len(plot_data),
    )


def export_cost_table() -> None:
    coverage = pd.read_excel(COST_XLSX, sheet_name="Coverage", header=3)
    save_data(coverage, "complete_build_cost_coverage_W1_W4_S1_S8")


def write_status_files() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(STATUS).to_csv(OUT_DIR / "figure_status_20260918.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "generated_at": "2026-09-18",
        "output_directory": str(OUT_DIR),
        "figures": STATUS,
        "completion_rule": "all requested figures attempted; semantically unavailable panels are recorded as skipped",
    }
    (OUT_DIR / "figure_manifest_20260918.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    complete = all(item["status"] in {"complete", "skipped"} for item in STATUS)
    marker = OUT_DIR / "FIGURES_COMPLETE" if complete else OUT_DIR / "FIGURES_PARTIAL"
    other_marker = OUT_DIR / "FIGURES_PARTIAL" if complete else OUT_DIR / "FIGURES_COMPLETE"
    if other_marker.exists():
        other_marker.unlink()
    marker.write_text("generated\n", encoding="utf-8")


def run_one(name: str, function) -> None:
    try:
        function()
        print(f"[done] {name}")
    except Exception as exc:  # Continue with the next figure as requested.
        add_status(name, "all", "failed", "local/remote source audit", f"{type(exc).__name__}: {exc}")
        print(f"[failed {name}] {traceback.format_exc()}")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_one("5.6.1", run_existing_six_panel)
    run_one("5.6.2", draw_workload_awareness)
    run_one("5.6.2-bc", draw_range_pair)
    run_one("S1-S5-algorithms", draw_s1_s5_algorithm_pair)
    run_one("5.6.2-d-cc-news", draw_cc_news_correlation_pair)
    run_one("5.6.2-d-artificial", draw_artificial_correlation_pair)
    run_one("5.6.2-e-standalone", draw_expensive_predicate_performance)
    run_one("5.6.3", draw_thread_scaling)
    run_one("5.6.4", draw_stability)
    run_one("5.7-a", draw_build_cost_decomposition)
    run_one("5.7-b", draw_build_quality)
    try:
        export_cost_table()
    except Exception as exc:
        print(f"[cost table skipped] {type(exc).__name__}: {exc}")
    write_status_files()
    print(f"[output] {OUT_DIR}")
    print(pd.DataFrame(STATUS).to_string(index=False))


if __name__ == "__main__":
    main()
