import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "docs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_main_metrics():
    metrics = {
        "Fine exact": 57.86,
        "Fine near (+/-1)": 92.58,
        "Coarse age": 88.38,
        "Gender": 87.46,
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values)

    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Best Model Evaluation Metrics")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_model_metrics.png"), dpi=200)
    plt.close()


def plot_fine_age_per_class():
    labels = [
        "0-13",
        "14-17",
        "18-24",
        "25-33",
        "34-48",
        "49-64",
        "65+",
    ]

    precision = [0.85, 0.58, 0.44, 0.54, 0.46, 0.55, 0.68]
    recall = [0.90, 0.04, 0.16, 0.69, 0.52, 0.50, 0.68]
    f1 = [0.87, 0.07, 0.24, 0.60, 0.49, 0.53, 0.68]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(11, 5))

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Fine Age Per-Class Performance")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fine_age_per_class_scores.png"), dpi=200)
    plt.close()


def plot_experiment_comparison():
    experiments = [
        "WIKI+IMDB\n9-class",
        "7-class\nsoft labels",
        "UTK fine-tuned\n7-class",
    ]

    fine_exact = [48.65, 44.99, 57.86]
    fine_near = [88.62, 87.31, 92.58]

    x = np.arange(len(experiments))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(x - width / 2, fine_exact, width, label="Fine exact")
    plt.bar(x + width / 2, fine_near, width, label="Fine near (+/-1)")

    plt.xticks(x, experiments)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Experiment Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "experiment_comparison.png"), dpi=200)
    plt.close()


def save_summary_csv():
    df = pd.DataFrame([
        {
            "model": "Hierarchical EfficientNet + UTK fine-tune",
            "fine_exact_accuracy": 0.5786,
            "fine_near_accuracy": 0.9258,
            "coarse_accuracy": 0.8838,
            "gender_accuracy": 0.8746,
        }
    ])

    df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)


def main():
    plot_main_metrics()
    plot_fine_age_per_class()
    plot_experiment_comparison()
    save_summary_csv()

    print(f"Saved plots and summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()