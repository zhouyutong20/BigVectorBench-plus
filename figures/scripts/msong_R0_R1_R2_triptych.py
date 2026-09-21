"""Draw the MSong R0/R1/R2 stage-microbenchmark triptych.

R0 = no_filter, R1 = tight_12, and R2 = wide_80.
The source table contains one five-run mean per algorithm and condition,
so the figure intentionally uses points with QPS error bars instead of
connecting repeated measurements into artificial curves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = BASE_DIR.parent
SOURCE_XLSX = EXPERIMENT_DIR / "实验结果.xlsx"
OUTPUT_PNG = BASE_DIR / "msong_R0_R1_R2_triptych.png"
OUTPUT_PDF = BASE_DIR / "msong_R0_R1_R2_triptych.pdf"
OUTPUT_AUDIT = BASE_DIR / "msong_R0_R1_R2_triptych_audit.csv"


ALGORITHM_ORDER = [
    "faiss-hnsw",
    "hnswlib-hnsw",
    "milvus-hnsw",
    "weaviate-hnsw",
]

ALGORITHM_LABELS = {
    "faiss-hnsw": "Faiss-HNSW",
    "hnswlib-hnsw": "HNSWLib-HNSW",
    "milvus-hnsw": "Milvus-HNSW",
    "weaviate-hnsw": "Weaviate-HNSW",
}

COLORS = {
    "faiss-hnsw": "#1f77b4",
    "hnswlib-hnsw": "#ff7f0e",
    "milvus-hnsw": "#2ca02c",
    "weaviate-hnsw": "#d62728",
}

MARKERS = {
    "faiss-hnsw": "s",
    "hnswlib-hnsw": "^",
    "milvus-hnsw": "P",
    "weaviate-hnsw": "o",
}

CONDITION_ORDER = ["no_filter", "tight_12", "wide_80"]
PANEL_LABELS = {
    "no_filter": "(a) R0: No filter",
    "tight_12": "(b) R1: Tight",
    "wide_80": "(c) R2: Wide",
}


def format_qps(value, _position):
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:g}"


def load_stage_data():
    data = pd.read_excel(SOURCE_XLSX, sheet_name="W4_Stage_Agg")
    data.columns = [str(column).strip().lower() for column in data.columns]
    data = data[
        data["algorithm"].astype(str).str.lower().isin(ALGORITHM_ORDER)
        & data["condition"].astype(str).str.lower().isin(CONDITION_ORDER)
    ].copy()

    numeric_columns = ["qps_mean", "qps_std", "recall_mean", "recall_std", "runs"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["algorithm"] = data["algorithm"].str.lower()
    data["condition"] = data["condition"].str.lower()
    data["algorithm_label"] = data["algorithm"].map(ALGORITHM_LABELS)
    data["panel"] = data["condition"].map(PANEL_LABELS)
    data = data.sort_values(
        by=["condition", "algorithm"],
        key=lambda series: series.map(
            {value: index for index, value in enumerate(CONDITION_ORDER)}
            if series.name == "condition"
            else series.map({value: index for index, value in enumerate(ALGORITHM_ORDER)})
        ),
    )
    return data.reset_index(drop=True)


def draw_figure(data):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.27,
        top=0.76,
        wspace=0.27,
    )

    legend_handles = []
    legend_labels = []

    for column, condition in enumerate(CONDITION_ORDER):
        ax = axes[column]
        panel = data[data["condition"] == condition]

        for algorithm in ALGORITHM_ORDER:
            row = panel[panel["algorithm"] == algorithm]
            if row.empty:
                continue
            row = row.iloc[0]
            qps = float(row["qps_mean"])
            qps_std = float(row["qps_std"])
            recall = float(row["recall_mean"])

            lower = max(qps - qps_std, qps * 0.01)
            upper = qps + qps_std
            ax.errorbar(
                recall,
                qps,
                yerr=np.array([[qps - lower], [upper - qps]]),
                fmt="none",
                color=COLORS[algorithm],
                elinewidth=2.8,
                capsize=6,
                capthick=2.8,
                alpha=0.9,
                zorder=2,
            )
            handle = ax.scatter(
                recall,
                qps,
                s=245,
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                edgecolor="white",
                linewidth=1.4,
                alpha=0.97,
                zorder=3,
                label=ALGORITHM_LABELS[algorithm],
            )
            if column == 0:
                legend_handles.append(handle)
                legend_labels.append(ALGORITHM_LABELS[algorithm])

        ax.set_xlim(0.975, 1.0003)
        ax.set_xticks([0.98, 0.99, 1.00])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.2f}"))
        ax.set_yscale("log")
        ax.set_ylim(1, 10000)
        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.yaxis.set_major_formatter(FuncFormatter(format_qps))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())

        ax.set_xlabel("Recall", fontsize=32, fontweight="bold", labelpad=14)
        if column == 0:
            ax.set_ylabel("QPS", fontsize=32, fontweight="bold", labelpad=14)
        else:
            ax.set_ylabel("")
        ax.text(
            0.5,
            -0.31,
            PANEL_LABELS[condition],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=32,
            fontweight="bold",
            clip_on=False,
        )
        ax.tick_params(axis="both", which="major", labelsize=25, width=2.8, length=9, pad=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(2.8)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        prop={"size": 25, "weight": "bold"},
        columnspacing=1.25,
        handlelength=1.8,
        handletextpad=0.55,
    )

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    data = load_stage_data()
    if len(data) != len(ALGORITHM_ORDER) * len(CONDITION_ORDER):
        raise RuntimeError(
            f"Expected {len(ALGORITHM_ORDER) * len(CONDITION_ORDER)} rows, got {len(data)}"
        )

    audit_columns = [
        "condition",
        "panel",
        "algorithm_label",
        "recall_mean",
        "qps_mean",
        "qps_std",
        "runs",
    ]
    data[audit_columns].to_csv(OUTPUT_AUDIT, index=False, encoding="utf-8-sig")
    draw_figure(data)

    print("[source]", SOURCE_XLSX)
    print("[algorithms]", ", ".join(ALGORITHM_LABELS[value] for value in ALGORITHM_ORDER))
    print("[conditions] R0=no_filter, R1=tight_12, R2=wide_80")
    print("[query_workers] 28")
    print("[runs_per_point] 5")
    print("[saved]", OUTPUT_PNG)
    print("[saved]", OUTPUT_PDF)
    print("[audit]", OUTPUT_AUDIT)


if __name__ == "__main__":
    main()
