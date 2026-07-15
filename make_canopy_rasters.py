import argparse
import glob
import os
import shutil
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd

PRODUCTS = ("dtm", "dsm", "chm", "cc", "cbh", "cbd")
NODATA = np.float32(-9999.0)


def mosaic_product(tile_paths, output_path):
    sources = [rasterio.open(p) for p in sorted(tile_paths)]
    mosaic, mosaic_transform = merge(sources, nodata=float(NODATA), method="first")
    profile = sources[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        compress="lzw",
        nodata=float(NODATA),
    )
    for src in sources:
        src.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
    print(f"  Mosaic: {mosaic.shape[2]}×{mosaic.shape[1]} cells → {output_path}")


def clip_to_aoi(input_path, output_path, shapes):
    with rasterio.open(input_path) as src:
        out_image, out_transform = rasterio_mask(
            src, shapes, crop=True, nodata=float(NODATA)
        )
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="lzw",
        )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_image)
    valid = out_image[0] != float(NODATA)
    print(f"  Clipped: {valid.sum():,} valid cells → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mosaic per-tile rasters → final clipped canopy products"
    )
    parser.add_argument("--tiles_dir", default="data/processed/tiles",
                        help="Directory containing per-tile GeoTIFFs from classify_ground.py")
    parser.add_argument("--output_dir", default="data/processed",
                        help="Directory for final clipped products")
    parser.add_argument("--perimeter", required=True,
                        help="Fire perimeter GeoJSON for AOI clipping")
    parser.add_argument("--buffer", type=float, default=1000.0,
                        help="Buffer around perimeter in meters (default: 1000)")
    parser.add_argument("--fire_name", default="airplane_lake",
                        help="Prefix for output filenames")
    args = parser.parse_args()

    print(f"Loading fire perimeter: {args.perimeter}")
    perimeter = gpd.read_file(args.perimeter).to_crs(epsg=32610)
    buffered = perimeter.buffer(args.buffer)
    shapes = [geom.__geo_interface__ for geom in buffered.geometry]

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_mosaic_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for product in PRODUCTS:
        print(f"\n--- {product.upper()} ---")
        tile_paths = glob.glob(os.path.join(args.tiles_dir, f"*_{product}.tif"))
        if not tile_paths:
            print(f"  No tiles found — skipping (run classify_ground.py first)")
            continue

        print(f"  Mosaicking {len(tile_paths)} tiles...")
        mosaic_path = os.path.join(tmp_dir, f"{product}_mosaic.tif")
        mosaic_product(tile_paths, mosaic_path)

        final_path = os.path.join(args.output_dir, f"{args.fire_name}_{product}_10m.tif")
        clip_to_aoi(mosaic_path, final_path, shapes)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nDone. Final products in {args.output_dir}/")
    print("  airplane_lake_dtm_10m.tif  — bare-earth elevation (m above sea level)")
    print("  airplane_lake_dsm_10m.tif  — first-return surface elevation")
    print("  airplane_lake_chm_10m.tif  — canopy height above ground (m), clipped [0,100]")
    print("  airplane_lake_cc_10m.tif   — canopy cover (% first returns > 2m height)")
    print("  airplane_lake_cbh_10m.tif  — canopy base height (m), from return-density gap")
    print("  airplane_lake_cbd_10m.tif  — canopy bulk density proxy (kg/m^3, placeholder scale)")


if __name__ == "__main__":
    main()
