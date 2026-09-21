from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter, NullLocator


OUT_DIR = Path(
    r"<local_workspace>\新增实验\算法库的数据画图\figures_5_6_5_7_20260918"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUT_DIR / "fig_5_6_2_acorn_single_thread_c1_c4.png"
OUTPUT_CSV = OUT_DIR / "fig_5_6_2_acorn_single_thread_c1_c4_data.csv"


DATASETS = [
    {
        "code": "C1",
        "name": "Artificial-corr-12 original",
        "rows": [
            (100, 0.8587, 193.1),
            (200, 0.9295, 112.2),
            (400, 0.9727, 66.3),
            (800, 0.9917, 39.4),
            (1600, 0.9980, 29.7),
        ],
    },
    {
        "code": "C2",
        "name": "Artificial-corr-12 shuffled",
        "rows": [
            (100, 0.9657, 243.3),
            (200, 0.9888, 140.5),
            (400, 0.9970, 80.6),
            (800, 0.9993, 45.1),
            (1600, 0.9999, 27.8),
        ],
    },
    {
        "code": "C3",
        "name": "CC-News original",
        "rows": [
            (100, 0.8649, 392.6),
            (200, 0.9010, 217.8),
            (400, 0.9169, 119.9),
            (800, 0.9238, 62.3),
            (1600, 0.9262, 31.0),
        ],
    },
    {
        "code": "C4",
        "name": "CC-News shuffled",
        "rows": [
            (100, 0.8301, 351.2),
            (200, 0.8742, 195.5),
            (400, 0.8954, 105.3),
            (800, 0.9041, 54.9),
            (1600, 0.9073, 27.6),
        ],
    },
]


def save_data() -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "dataset_name", "efSearch", "recall_at_100", "qps", "threads"])
        for dataset in DATASETS:
            for ef_search, recall, qps in dataset["rows"]:
                writer.writerow(
                    [dataset["code"], dataset["name"], ef_search, f"{recall:.4f}", f"{qps:.1f}", 1]
                )


def draw() -> None:
    save_data()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": 17,
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "axes.linewidth": 1.8,
            "legend.fontsize": 19,
            "legend.frameon": False,
            "savefig.dpi": 400,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()

    color = "#377eb8"
    marker = "o"
    for ax, dataset in zip(axes, DATASETS):
        rows = dataset["rows"]
        ef_values = [row[0] for row in rows]
        recalls = [row[1] for row in rows]
        qps = [row[2] for row in rows]

        ax.plot(
            recalls,
            qps,
            color=color,
            linewidth=3.0,
            marker=marker,
            markersize=10.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label="ACORN (1 thread)",
            zorder=3,
        )

        # The parameter order is encoded by the connected curve; annotate only
        # the endpoints to keep the four-panel figure readable.
        ax.annotate(
            f"ef={ef_values[0]}",
            (recalls[0], qps[0]),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color="#3d3d3d",
        )
        ax.annotate(
            f"ef={ef_values[-1]}",
            (recalls[-1], qps[-1]),
            xytext=(-54, -18),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            color="#3d3d3d",
        )

        ax.set_title(f"{dataset['code']}: {dataset['name']}", pad=10)
        ax.set_yscale("log")
        ax.set_xlim(0.82, 1.01)
        ax.set_ylim(20, 500)
        ax.xaxis.set_major_locator(FixedLocator([0.85, 0.90, 0.95, 1.00]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.28)
        ax.grid(axis="x", which="major", linestyle=":", linewidth=0.7, alpha=0.18)
        ax.tick_params(axis="both", which="major", length=6, width=1.6)
        ax.set_box_aspect(0.75)  # 4:3 axes box, matching the paper template.

    for ax in axes[2:]:
        ax.set_xlabel("Recall@100", labelpad=8)
    for ax in axes[::2]:
        ax.set_ylabel("QPS", labelpad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=1,
        handlelength=2.4,
        handletextpad=0.7,
        borderaxespad=0.0,
        prop={"size": 20, "weight": "bold"},
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.845,
        wspace=0.14,
        hspace=0.31,
    )
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    draw()
