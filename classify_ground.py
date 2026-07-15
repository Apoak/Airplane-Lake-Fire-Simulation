import argparse
import glob
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import geopandas as gpd
import laspy
from scipy.ndimage import distance_transform_edt

CELL = 10.0          # 10m output resolution
OUTPUT_CRS = "EPSG:32610"
NODATA = np.float32(-9999.0)
HEIGHT_THRESHOLD = 2.0   # meters above ground for canopy cover
MIN_CC_RETURNS = 3        # min first returns needed for a valid CC cell

# --- CBH/CBD vertical profile parameters ---
PROFILE_BIN = 1.0          # vertical bin size (m) for the return-density profile
PROFILE_MAX_HEIGHT = 60.0  # profile ceiling; taller returns fold into the top bin
PROFILE_MIN_HEIGHT = 1.0   # ignore returns below this (ground vegetation/shrub noise)
CBH_REL_DENSITY = 0.05     # bin counts as "canopy" if density > 5% of the column's peak bin
CBH_MIN_COUNT = 3          # ...and at least this many returns (guards sparse/noisy columns)
MIN_VEG_RETURNS = 5        # min non-ground returns needed for a valid CBH/CBD cell
CBD_SCALE = 0.05           # kg/m^3 per (return/m^3); rough placeholder — see Day 5 calibration
MIN_CANOPY_DEPTH = 2.0     # floor (m) on the CBD denominator; thin crowns otherwise blow up
CBD_MAX = 0.5              # kg/m^3 plausibility clip (typical conifer CBD tops out ~0.3-0.4)


def get_tile_bbox(laz_path):
    """Read bounding box from LAZ header without decompressing points."""
    with laspy.open(laz_path) as f:
        mins = f.header.mins
        maxs = f.header.maxs
    return mins[0], mins[1], maxs[0], maxs[1]


def bbox_intersects(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)


def write_raster(path, data, transform, crs, nodata=NODATA):
    out = np.where(np.isnan(data), nodata, data).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=np.float32, crs=crs,
        transform=transform, nodata=float(nodata),
        compress="lzw",
    ) as dst:
        dst.write(out[np.newaxis, :, :])


def classify_ground_pmf(x, y, z, cell_size=1.0):
    """
    Progressive Morphological Filter (Zhang et al. 2003).
    Fallback for tiles without ASPRS class-2 ground classification.

    Concept: rolling a ball across the underside of the point cloud. Wherever
    the ball fits, it marks ground level. Trees (narrow vertically) get rolled
    over; broad terrain features survive. Larger balls remove larger vegetation.
    """
    from scipy.ndimage import grey_opening

    xmin_p, ymax_p = x.min(), y.max()
    ncols = int(np.ceil((x.max() - xmin_p) / cell_size)) + 1
    nrows = int(np.ceil((ymax_p - y.min()) / cell_size)) + 1

    col_p = np.clip(np.floor((x - xmin_p) / cell_size).astype(np.int32), 0, ncols - 1)
    row_p = np.clip(np.floor((ymax_p - y) / cell_size).astype(np.int32), 0, nrows - 1)
    idx_p = row_p * ncols + col_p

    min_z = np.full(nrows * ncols, np.inf, dtype=np.float32)
    np.minimum.at(min_z, idx_p, z)
    min_z[min_z == np.inf] = np.nan
    min_z = min_z.reshape(nrows, ncols)

    valid = ~np.isnan(min_z)
    _, nn_idx = distance_transform_edt(~valid, return_indices=True)
    surface = min_z[tuple(nn_idx)]

    # Progressively open with growing kernels (3m → 33m at 1m cells)
    # Each pass removes vegetation features smaller than the kernel radius.
    opened = surface.copy()
    for ksize in [3, 5, 9, 17, 33]:
        opened = np.minimum(opened, grey_opening(surface, size=(ksize, ksize)))

    ground_surface_at_point = opened[row_p, col_p]
    return (z - ground_surface_at_point) <= 0.5


def process_tile(laz_path, output_dir):
    tile_name = Path(laz_path).stem
    out_paths = {
        product: os.path.join(output_dir, f"{tile_name}_{product}.tif")
        for product in ("dtm", "dsm", "chm", "cc", "cbh", "cbd")
    }

    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in out_paths.values()):
        print(f"  Skipping {tile_name} (all outputs exist)")
        return

    print(f"  Processing {tile_name} ...")

    las = laspy.read(laz_path)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float32)
    classification = np.array(las.classification, dtype=np.uint8)
    return_number = np.array(las.return_number, dtype=np.uint8)

    if 2 in np.unique(classification):
        # Use USGS-provided ground classification (expected for 3DEP LPC products)
        ground_mask = (classification == 2)
    else:
        print(f"    WARNING: no class-2 points — running PMF fallback")
        ground_mask = classify_ground_pmf(x, y, z)

    ground_pct = 100.0 * ground_mask.sum() / len(z)
    print(f"    Ground returns: {ground_mask.sum():,} ({ground_pct:.1f}%)")

    xmin = las.header.mins[0]
    ymin_h = las.header.mins[1]
    xmax = las.header.maxs[0]
    ymax = las.header.maxs[1]
    NCOLS = max(1, int(round((xmax - xmin) / CELL)))
    NROWS = max(1, int(round((ymax - ymin_h) / CELL)))

    col = np.clip(np.floor((x - xmin) / CELL).astype(np.int32), 0, NCOLS - 1)
    row = np.clip(np.floor((ymax - y) / CELL).astype(np.int32), 0, NROWS - 1)
    cell_idx = row * NCOLS + col

    transform = from_origin(xmin, ymax, CELL, CELL)

    # --- DTM: minimum Z of ground returns per cell ---
    dtm_flat = np.full(NROWS * NCOLS, np.inf, dtype=np.float32)
    np.minimum.at(dtm_flat, cell_idx[ground_mask], z[ground_mask])
    dtm_flat[dtm_flat == np.inf] = np.nan
    dtm = dtm_flat.reshape(NROWS, NCOLS)

    # Gap-fill: cells with no ground returns get the nearest valid cell's value.
    n_nan = int(np.isnan(dtm).sum())
    if n_nan > 0:
        fill_pct = 100.0 * n_nan / dtm.size
        if fill_pct > 20:
            print(f"    NOTE: {fill_pct:.1f}% of DTM cells gap-filled (data void or low density)")
        _, nn_idx = distance_transform_edt(np.isnan(dtm), return_indices=True)
        dtm_filled = dtm[tuple(nn_idx)]
    else:
        dtm_filled = dtm

    write_raster(out_paths["dtm"], dtm_filled, transform, OUTPUT_CRS)

    # --- DSM: maximum Z of first returns per cell ---
    # Exclude class 7 (low noise) and class 18 (high noise, ASPRS LAS 1.4)
    first_mask = (return_number == 1) & (classification != 7) & (classification != 18)
    dsm_flat = np.full(NROWS * NCOLS, -np.inf, dtype=np.float32)
    np.maximum.at(dsm_flat, cell_idx[first_mask], z[first_mask])
    dsm_flat[dsm_flat == -np.inf] = np.nan
    dsm = dsm_flat.reshape(NROWS, NCOLS)
    write_raster(out_paths["dsm"], dsm, transform, OUTPUT_CRS)

    # --- CHM: canopy height = DSM - ground surface, clipped [0, 100m] ---
    chm = np.clip(dsm - dtm_filled, 0.0, 100.0)
    chm[np.isnan(dsm)] = np.nan
    write_raster(out_paths["chm"], chm, transform, OUTPUT_CRS)

    # --- CC: fraction of first returns above HEIGHT_THRESHOLD per cell ---
    # Height above ground, not above sea level.
    ground_at_pt = dtm_filled[row[first_mask], col[first_mask]]
    height_above_ground = z[first_mask] - ground_at_pt

    total_first = np.zeros(NROWS * NCOLS, dtype=np.float32)
    above_thresh = np.zeros(NROWS * NCOLS, dtype=np.float32)
    np.add.at(total_first, cell_idx[first_mask], 1.0)
    np.add.at(above_thresh, cell_idx[first_mask],
               (height_above_ground > HEIGHT_THRESHOLD).astype(np.float32))

    with np.errstate(invalid="ignore", divide="ignore"):
        cc_raw = above_thresh / total_first * 100.0
    cc_flat = np.where(total_first >= MIN_CC_RETURNS, cc_raw, np.nan)
    cc = cc_flat.reshape(NROWS, NCOLS)
    write_raster(out_paths["cc"], cc, transform, OUTPUT_CRS)

    # --- CBH: canopy base height, from the vertical return-density profile ---
    # Bin all non-ground, non-noise returns by height-above-ground per cell, then
    # walk down from the canopy top until the density drops below threshold — the
    # first such gap marks the top of the understory / bottom of the live crown.
    veg_mask = (classification != 2) & (classification != 7) & (classification != 18)
    ground_at_veg = dtm_filled[row[veg_mask], col[veg_mask]]
    veg_height = np.clip(z[veg_mask] - ground_at_veg, 0.0, PROFILE_MAX_HEIGHT)
    veg_cell = cell_idx[veg_mask]

    n_bins = int(PROFILE_MAX_HEIGHT / PROFILE_BIN)
    min_bin = int(PROFILE_MIN_HEIGHT / PROFILE_BIN)
    bin_idx = np.clip(np.floor(veg_height / PROFILE_BIN).astype(np.int32), 0, n_bins - 1)

    profile_flat = np.zeros(NROWS * NCOLS * n_bins, dtype=np.float32)
    np.add.at(profile_flat, veg_cell * n_bins + bin_idx, 1.0)
    profile = profile_flat.reshape(NROWS * NCOLS, n_bins)

    col_max = profile.max(axis=1, keepdims=True)
    is_canopy = profile > np.maximum(col_max * CBH_REL_DENSITY, CBH_MIN_COUNT)
    is_canopy[:, :min_bin] = False  # never report a crown base below PROFILE_MIN_HEIGHT

    # The profile grid has a fixed ceiling (PROFILE_MAX_HEIGHT), but most trees are
    # shorter than that — so the top-down scan must start at each cell's actual
    # highest occupied bin, not the fixed array end. Treat bins above that point as
    # nonexistent headroom (force True) so they can't masquerade as a density gap,
    # and force the top bin itself into the canopy (it's the treetop, by definition,
    # however few returns hit it).
    has_return = profile > 0
    any_return = has_return.any(axis=1)
    top_bin = n_bins - 1 - np.argmax(has_return[:, ::-1], axis=1)
    bin_positions = np.arange(n_bins)[np.newaxis, :]
    is_canopy_adj = is_canopy | (bin_positions > top_bin[:, np.newaxis])
    is_canopy_adj[np.arange(NROWS * NCOLS), top_bin] = True

    contiguous_from_top = np.cumprod(is_canopy_adj[:, ::-1], axis=1).sum(axis=1)
    cbh_bin = n_bins - contiguous_from_top
    cbh_flat = (cbh_bin * PROFILE_BIN).astype(np.float32)

    veg_count = np.zeros(NROWS * NCOLS, dtype=np.float32)
    np.add.at(veg_count, veg_cell, 1.0)
    cbh_flat[~any_return | (top_bin < min_bin) | (veg_count < MIN_VEG_RETURNS)] = np.nan
    cbh_flat[np.isnan(chm.ravel())] = np.nan
    cbh = cbh_flat.reshape(NROWS, NCOLS)
    write_raster(out_paths["cbh"], cbh, transform, OUTPUT_CRS)

    # --- CBD: canopy bulk density proxy — return density within [CBH, CHM] layer ---
    # LiDAR return density in the crown layer is a proxy for foliage volume, not a
    # direct mass measurement. CBD_SCALE is a placeholder converting it to a
    # plausible kg/m^3 range; Day 5 calibration against LANDFIRE should refine it.
    cbh_at_pt = cbh_flat[veg_cell]
    canopy_layer = veg_height > cbh_at_pt
    canopy_count = np.zeros(NROWS * NCOLS, dtype=np.float32)
    np.add.at(canopy_count, veg_cell[canopy_layer], 1.0)

    canopy_depth = (chm.ravel() - cbh_flat)
    # Floor the denominator — a cell with a real but very thin crown (or a stray
    # point right at CHM) otherwise divides by a near-zero depth and blows up.
    cell_volume = CELL * CELL * np.maximum(canopy_depth, MIN_CANOPY_DEPTH)
    with np.errstate(invalid="ignore", divide="ignore"):
        return_density = canopy_count / cell_volume
    cbd_flat = np.clip(return_density * CBD_SCALE, 0.0, CBD_MAX).astype(np.float32)
    cbd_flat[(canopy_depth <= 0) | np.isnan(cbh_flat)] = np.nan
    cbd = cbd_flat.reshape(NROWS, NCOLS)
    write_raster(out_paths["cbd"], cbd, transform, OUTPUT_CRS)

    print(f"    DTM [{dtm_filled.min():.0f}–{dtm_filled.max():.0f}m], "
          f"CHM max={np.nanmax(chm):.1f}m, CC mean={np.nanmean(cc):.1f}%, "
          f"CBH mean={np.nanmean(cbh):.1f}m, CBD mean={np.nanmean(cbd):.3f}kg/m3")


def main():
    parser = argparse.ArgumentParser(
        description="Process LiDAR tiles → per-tile DTM, DSM, CHM, CC rasters at 10m"
    )
    parser.add_argument("--laz_dir", default="laz_files",
                        help="Directory containing LAZ files")
    parser.add_argument("--perimeter", required=True,
                        help="Fire perimeter GeoJSON (used to select overlapping tiles)")
    parser.add_argument("--output_dir", default="data/processed/tiles",
                        help="Directory for per-tile output rasters")
    parser.add_argument("--buffer", type=float, default=1000.0,
                        help="Buffer around perimeter in meters (default: 1000)")
    args = parser.parse_args()

    print(f"Loading fire perimeter: {args.perimeter}")
    perimeter = gpd.read_file(args.perimeter).to_crs(epsg=32610)
    aoi_bounds = tuple(perimeter.buffer(args.buffer).total_bounds)
    print(f"AOI bounds (UTM 10N, {args.buffer}m buffer): "
          f"X={aoi_bounds[0]:.0f}–{aoi_bounds[2]:.0f}, Y={aoi_bounds[1]:.0f}–{aoi_bounds[3]:.0f}")

    all_laz = sorted(glob.glob(os.path.join(args.laz_dir, "*.laz")))
    all_laz = [p for p in all_laz if not p.endswith(".copc.laz")]
    print(f"Found {len(all_laz)} LAZ tiles (excluding .copc.laz)")

    selected = [p for p in all_laz
                if bbox_intersects(*get_tile_bbox(p), *aoi_bounds)]
    print(f"Tiles overlapping AOI: {len(selected)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, laz_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {os.path.basename(laz_path)}")
        try:
            process_tile(laz_path, args.output_dir)
        except Exception as e:
            print(f"  ERROR processing {laz_path}: {e}")
            raise


if __name__ == "__main__":
    main()
