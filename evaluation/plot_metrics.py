
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("evaluation/results/retrieval_summary.csv")
OUTPUT = Path("evaluation/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS)

print("Loaded evaluation results:")
print(df.to_string(index=False))

df = df.set_index("method")

ks = [1, 3, 5]

recall = pd.DataFrame(
    {
        method: [
            df.loc[method, f"recall_at_{k}"]
            for k in ks
        ]
        for method in df.index
    },
    index=ks,
)

mrr = pd.DataFrame(
    {
        method: [
            df.loc[method, f"rr_at_{k}"]
            for k in ks
        ]
        for method in df.index
    },
    index=ks,
)

for filename, data, ylabel in [
    ("recall_comparison", recall, "Recall@K"),
    ("mrr_comparison", mrr, "MRR@K"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in data.columns:
        ax.plot(
            data.index,
            data[method],
            marker="o",
            linewidth=2,
            markersize=8,
            label=method,
        )

    ax.set_title(
        f"DocuMind AI - {ylabel} Comparison",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("K (Retrieved Documents)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT / f"{filename}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")

print("\nAll evaluation charts generated successfully!")
