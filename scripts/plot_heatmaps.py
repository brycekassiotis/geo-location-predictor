from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'granularity_study'

RUNS = [
    ("Coarse: 87 cells", "q2500"),
    ("Medium: 407 cells", "q500_main"),
    ("Fine: 1,841 cells", "q100"),
]

BIN_SIZE = 5
MIN_IMAGES = 1

def calculate_grid(df):
    lat_edges = np.arange(-90, 91, BIN_SIZE)
    lon_edges = np.arange(-180, 181, BIN_SIZE)

    df = df.copy()

    df["lat_bin"] = np.clip(np.searchsorted(lat_edges, df["true_lat"], side="right") - 1, 0, len(lat_edges) - 2,)

    df["lon_bin"] = np.clip(np.searchsorted(lon_edges, df["true_lon"], side="right") - 1, 0, len(lon_edges) - 2,)

    grouped = (
        df.groupby(["lat_bin", "lon_bin"])["dist_km"]
        .agg(["median", "count"])
        .reset_index()
    )

    grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)

    for row in grouped.itertuples():
        if row.count >= MIN_IMAGES:
            grid[int(row.lat_bin), int(row.lon_bin)] = row.median

    return grid, lon_edges, lat_edges, grouped

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = []

    for title, folder in RUNS:
        df = pd.read_csv(ROOT / "results" / folder / "preds_val.csv", dtype={"id": str})

        required = df[["true_lat", "true_lon", "dist_km"]]

        if not np.isfinite(required.to_numpy()).all():
            raise ValueError(f"Invalid values in {folder}.")

        grid, lon_edges, lat_edges, grouped = calculate_grid(df)

        grouped.to_csv(OUTPUT / f"heatmap_bins_{folder}.csv", index=False)

        results.append((title, grid, lon_edges, lat_edges))

    valid_values = np.concatenate([grid[np.isfinite(grid)] for _, grid, _, _ in results])

    if len(valid_values) == 0:
        raise ValueError(
            "No bins meet MIN_IMAGES. Reduce MIN_IMAGES."
        )

    positive = valid_values[valid_values > 0]
    vmin = max(float(positive.min()), 0.01) if len(positive) else 0.01
    vmax = max(float(valid_values.max()), vmin * 1.01)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), layout="constrained",)

    for ax, (title, grid, lon_edges, lat_edges) in zip(axes, results):
        
        masked = np.ma.masked_invalid(np.where(np.isfinite(grid), np.maximum(grid, vmin), grid))

        heatmap = ax.pcolormesh(
            lon_edges,
            lat_edges,
            masked,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )

        ax.set_title(title)
        ax.set_xlabel("Actual longitude (degrees)")
        ax.set_ylabel("Actual latitude (degrees)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(alpha=0.2)

    colorbar = fig.colorbar(
        heatmap,
        ax=axes,
        shrink=0.8,
        pad=0.02,
    )

    colorbar.set_label("Median location error (km; logarithmic scale)")

    fig.suptitle(
        "Geographic distribution of validation error\n"
        f"{BIN_SIZE}° bins; gray = fewer than {MIN_IMAGES} images", fontsize=14,)

    path = OUTPUT / "geographic_error_heatmap.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved heatmap: {path}")


if __name__ == "__main__":
    main()
