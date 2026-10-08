import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'granularity_study'

RUNS = [
    ('Coarse','q2500', 87),
    ('Medium', 'q500_main', 407),
    ('Fine', 'q100', 1841),
]

def load_runs():
    predictions = {}
    settings = []
    rows = []
    references = None

    for name, folder, cells in RUNS:
        run_path = ROOT / 'results' / folder

        df = pd.read_csv(run_path / 'preds_val.csv', dtype={'id': str})

        with open(run_path / 'metrics.json') as file:
            metadata = json.load(file)

        if df["id"].duplicated().any():
            raise ValueError(f"Duplicate validation IDs in {name} run")

        df = df.set_index('id').sort_index()

        if references is None:
            references = df
        else:
            if not df.index.equals(references.index):
                raise ValueError(f"Validation IDs differ in {name} run")
            if not np.allclose(df[['true_lat', 'true_lon']], references[['true_lat', 'true_lon']]):
                raise ValueError(f"Ground truth differs in {name} run")

        if metadata['num_cells'] != cells:
            raise ValueError(f"Unexpected cell count in {name} run")

        distances = df["dist_km"].to_numpy()

        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError(f"Nonfinite or negative prediction values in {name} run")

        row = {
            'granularity': name,
            'num_cells': cells,
            'validation_images': len(df),
            'median_km': np.median(distances),
            'mean_km': np.mean(distances),
            'geoscore': np.mean(5000 * np.exp(-distances / 1492.7)),
        }
        
        for threshold in (1, 25, 200, 750, 2500):
            row[f'acc@{threshold}km'] = np.mean(distances <= threshold)

        for metric, expected in metadata['best_val'].items():
            if not np.isclose(row[metric], expected, rtol=1e-5, atol=0.01):
                raise ValueError(f"Metrics mismatch for {metric} in {name} run")

        settings.append(metadata['train'])
        predictions[name] = df
        rows.append(row)

    if not all(setting == settings[0] for setting in settings):
        raise ValueError("Training settings differ across runs")

    return pd.DataFrame(rows), predictions

def plot_comparison(summary):
    colors = ['#52738c', '#198879', '#ce7740']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout='constrained',)

    metrics = [
        ("median_km", "Median error (km)", False),
        ("geoscore", "GeoScore", False),
        ("acc@25km", "Predictions within 25 km (%)", True),
    ]

    for ax, (column, title, percentage) in zip(axes, metrics):
        values = summary[column].to_numpy()

        bars = ax.bar(summary.granularity, values, color=colors)

        ax.bar_label(bars, fmt='%.2f', padding=4)
        ax.set_title(title)

        ax.set_ylim(0, max(values) * 1.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle('Effect of Geocell granularity')
    fig.savefig(OUTPUT / 'granularity_comparison.png', dpi=200)
    plt.close(fig)

def plot_error_distribution(predictions):
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')

    for name, df in predictions.items():
        distances = np.sort(df["dist_km"].to_numpy())

        percentages = (np.arange(1, len(distances) + 1) / len(distances) * 100)

        ax.step(np.maximum(distances, 0.01), percentages, where='post', label=name)

    ax.set_xscale('log')
    ax.set_xlabel('Location error (km; logarithmic scale)')
    ax.set_ylabel('Validation predictions within distance (%)')
    ax.set_title('Cumulative location error')
    ax.legend()
    ax.grid(alpha=0.25)

    fig.savefig(OUTPUT / 'error_distribution.png', dpi=200)
    plt.close(fig)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary, predictions = load_runs()

    summary.to_csv(OUTPUT / 'granularity_comparison.csv', index=False)

    plot_comparison(summary)
    plot_error_distribution(predictions)

    display = summary.copy()

    for column in display.columns:
        if column.startswith("acc@"):
            display[column] = display[column] * 100

    print("\nValidation comparison (accuracy columns are percentages):")
    print(display.round(2).to_string(index=False))

    print("\nVerified matching validation images and training settings.")
    print(f"Saved results to: {OUTPUT}")

if __name__ == "__main__":
    main()