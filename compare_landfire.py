import argparse
import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

NODATA = np.float32(-9999.0)
LANDFIRE_NODATA = 32767

# variable -> (LANDFIRE dir/file stem, LiDAR raster stem, scale factor, unit, display range)
VARIABLES = {
    "cc":  ("CC",  "cc",  1.0,   "%",     (0, 100)),
    "ch":  ("CH",  "chm", 1.0 / 10.0, "m",     (0, 60)),
    "cbh": ("CBH", "cbh", 1.0 / 10.0, "m",     (0, 40)),
    "cbd": ("CBD", "cbd", 1.0 / 100.0, "kg/m3", (0, 0.5)),
}

LANDFIRE_DIR = "/home/apokerlu/projects/pyrologix/LandFire"


def landfire_path(stem):
    return os.path.join(LANDFIRE_DIR, f"LF2022_{stem}_CONUS", f"LF2022_{stem}_CONUS.tif")


def reproject_landfire(var, ref_profile, out_dir):
    lf_stem, _, scale, unit, _ = VARIABLES[var]
    src_path = landfire_path(lf_stem)

    with rasterio.open(src_path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]), NODATA, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=LANDFIRE_NODATA,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=float(NODATA),
            resampling=Resampling.bilinear,
        )

    valid = dst != NODATA
    dst[valid] = dst[valid] * scale

    print(f"  [{var}] reprojected LANDFIRE {lf_stem}: shape={dst.shape}, "
          f"valid={valid.sum():,}/{valid.size:,} "
          f"min={dst[valid].min():.3f} max={dst[valid].max():.3f} mean={dst[valid].mean():.3f} ({unit})")

    out_path = os.path.join(out_dir, f"landfire_{var}_10m.tif")
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=float(NODATA), compress="lzw")
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(dst, 1)

    return dst, out_path


def load_lidar(var, processed_dir, fire_name):
    _, lidar_stem, _, _, _ = VARIABLES[var]
    path = os.path.join(processed_dir, f"{fire_name}_{lidar_stem}_10m.tif")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan
    return arr


def compute_stats(var, lidar, landfire):
    valid = np.isfinite(lidar) & (landfire != NODATA) & np.isfinite(landfire)
    n_valid = int(valid.sum())
    pct_coverage = 100.0 * n_valid / lidar.size

    l = lidar[valid]
    f = landfire[valid]
    diff = l - f
    bias = float(diff.mean())
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    r, _ = pearsonr(l, f) if n_valid > 2 else (np.nan, np.nan)

    stats = {
        "variable": var,
        "n_valid": n_valid,
        "pct_coverage": pct_coverage,
        "lidar_mean": float(l.mean()),
        "landfire_mean": float(f.mean()),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r),
    }

    diff_full = np.full_like(lidar, np.nan, dtype=np.float32)
    diff_full[valid] = diff
    return stats, diff_full, valid


def write_diff_raster(diff_full, ref_profile, out_path):
    arr = diff_full.copy()
    arr[~np.isfinite(arr)] = NODATA
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=float(NODATA), compress="lzw")
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(arr, 1)


def write_stats_md(all_stats, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "# Day 5 — LiDAR vs. LANDFIRE Agreement Stats\n",
        "| Variable | Unit | Valid cells | AOI coverage | LiDAR mean | LANDFIRE mean | Bias (LiDAR-LF) | MAE | RMSE | Pearson r |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in all_stats:
        unit = VARIABLES[s["variable"]][3]
        lines.append(
            f"| {s['variable'].upper()} | {unit} | {s['n_valid']:,} | {s['pct_coverage']:.1f}% | "
            f"{s['lidar_mean']:.3f} | {s['landfire_mean']:.3f} | {s['bias']:+.3f} | "
            f"{s['mae']:.3f} | {s['rmse']:.3f} | {s['pearson_r']:.3f} |"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nStats table written to {out_path}")
    print("\n".join(lines))


def plot_comparison_maps(data, out_path):
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    col_titles = ["LiDAR-derived", "LANDFIRE", "Diff (LiDAR - LANDFIRE)"]

    for row, var in enumerate(["cc", "ch", "cbh", "cbd"]):
        lidar = data[var]["lidar"]
        landfire = data[var]["landfire"]
        diff = data[var]["diff"]
        unit = VARIABLES[var][3]
        vmin, vmax = VARIABLES[var][4]

        for col, (arr, title) in enumerate(zip([lidar, landfire, diff], col_titles)):
            ax = axes[row, col]
            if col < 2:
                im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
            else:
                dmax = np.nanpercentile(np.abs(arr), 98) if np.isfinite(arr).any() else 1.0
                im = ax.imshow(arr, cmap="RdBu_r", norm=mcolors.CenteredNorm(vcenter=0, halfrange=max(dmax, 1e-6)))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{var.upper()} ({unit})", fontsize=12)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("LiDAR-Derived Canopy Fuel Metrics vs. LANDFIRE — Airplane Lake Fire AOI", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_scatter(data, all_stats, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    stats_by_var = {s["variable"]: s for s in all_stats}

    for ax, var in zip(axes.flat, ["cc", "ch", "cbh", "cbd"]):
        valid = data[var]["valid"]
        lidar = data[var]["lidar"][valid]
        landfire = data[var]["landfire"][valid]
        unit = VARIABLES[var][3]
        vmin, vmax = VARIABLES[var][4]

        hb = ax.hexbin(landfire, lidar, gridsize=40, cmap="viridis", mincnt=1, extent=(vmin, vmax, vmin, vmax))
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1, label="1:1")
        s = stats_by_var[var]
        ax.set_xlabel(f"LANDFIRE {var.upper()} ({unit})")
        ax.set_ylabel(f"LiDAR {var.upper()} ({unit})")
        ax.set_title(f"{var.upper()}: r={s['pearson_r']:.2f}, RMSE={s['rmse']:.2f}, n={s['n_valid']:,}")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="cell count")

    fig.suptitle("LiDAR vs. LANDFIRE Canopy Metric Correlation", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare LiDAR-derived canopy fuel rasters against LANDFIRE")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--fire_name", default="airplane_lake")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    ref_path = os.path.join(args.processed_dir, f"{args.fire_name}_chm_10m.tif")
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()

    print("=== Step 1: Reproject LANDFIRE layers onto LiDAR grid ===")
    landfire_arrays = {}
    for var in VARIABLES:
        arr, _ = reproject_landfire(var, ref_profile, args.processed_dir)
        landfire_arrays[var] = arr

    print("\n=== Step 2: Compute diffs and agreement stats ===")
    data = {}
    all_stats = []
    for var in VARIABLES:
        lidar = load_lidar(var, args.processed_dir, args.fire_name)
        landfire = landfire_arrays[var]
        stats, diff_full, valid = compute_stats(var, lidar, landfire)
        all_stats.append(stats)
        landfire_nan = np.where(landfire == NODATA, np.nan, landfire)
        data[var] = {"lidar": lidar, "landfire": landfire_nan, "diff": diff_full, "valid": valid}

        diff_path = os.path.join(args.processed_dir, f"diff_{var}_10m.tif")
        write_diff_raster(diff_full, ref_profile, diff_path)
        print(f"  [{var}] n_valid={stats['n_valid']:,} ({stats['pct_coverage']:.1f}% of AOI) "
              f"bias={stats['bias']:+.3f} rmse={stats['rmse']:.3f} r={stats['pearson_r']:.3f}")

    write_stats_md(all_stats, os.path.join(args.figures_dir, "day5_stats.md"))

    print("\n=== Step 3: Generate comparison figures ===")
    plot_comparison_maps(data, os.path.join(args.figures_dir, "day5_comparison_maps.png"))
    plot_scatter(data, all_stats, os.path.join(args.figures_dir, "day5_scatter_correlation.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
