# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A self-contained pipeline that derives canopy fuel structure from pre-fire LiDAR point clouds, compares those metrics against LANDFIRE's standard fuel layers, and runs Rothermel-based surface fire spread modeling — validated against the real perimeter of a historical Washington wildfire.

The intended pipeline stages are:
1. LiDAR point cloud → canopy height model, canopy cover, canopy base height, canopy bulk density
2. Compare LiDAR-derived canopy metrics against LANDFIRE's equivalent layers (CC, CH, CBH, CBD) for the same AOI
3. Implement Rothermel surface fire spread equations in Python (fuel model + terrain + weather → rate of spread, flame length rasters)
4. Simple fire growth simulation from an ignition point (cellular automaton or minimum-travel-time)
5. Validate simulated burn extent against the NIFC documented final perimeter

## Study Area & Data Sources

**Fire:** A historical Washington wildfire (~6,956 acres) with pre-fire USGS 3DEP LiDAR coverage and a documented final perimeter in the NIFC Interagency Fire Perimeter History dataset.

| Data | Source | Format |
|---|---|---|
| Fire perimeter | NIFC Interagency Fire Perimeter History | GeoJSON |
| LiDAR point cloud | USGS 3DEP (LidarExplorer) | LAZ (LPC product — not the pre-made DEM) |
| LANDFIRE layers | LANDFIRE LFPS / Data Access Tool | GeoTIFF |
| Base map (WA state) | CalTopo | MBTiles |

**LANDFIRE requires 8 layers, all from the same version/year (last version before the fire):**
`ELEV`, `SLPD`, `ASP`, `FBFM40` (primary Rothermel fuel input), `CC`, `CH`, `CBH`, `CBD`

LiDAR data for this project comes from two USGS surveys: the 2014 Glacier Peak QL1 survey and the 2019 Eastern Cascades survey.

## Technical Decisions

**CRS:** All layers must be reprojected to a single UTM CRS — **EPSG:32610 (UTM 10N)** for western WA (west of ~120°W) or **EPSG:32611 (UTM 11N)** for eastern WA. Never run analysis in EPSG:3857 or EPSG:4326.

**Resolution:** 10m, applied consistently to all rasters. LANDFIRE is natively 30m (resample up); LiDAR-derived layers resample down if finer.

**Resampling:** nearest-neighbor for categorical layers (FBFM40); bilinear or cubic for continuous layers (elevation, CC, CH, CBH, CBD).

**GeoTIFF editing:** Never use general image editors (Windows Photos, Paint, etc.) on GeoTIFFs — they strip georeferencing metadata. Use QGIS or `gdal_translate -projwin` instead.

## Known Gotchas

- **QGIS scratch outputs:** Processing tools default to temp files — always set an explicit output path or use Export → Save As.
- **LANDFIRE AOI submission:** The LFPS tool requires the AOI GeoJSON in **EPSG:3857** specifically. Verify coordinate magnitudes are large 7-digit meter values (not decimal degrees) before submitting.
- **GeoJSON multipart geometry:** QGIS buffer operations can output `MultiPolygon` even for a single polygon. Use "Multipart to Singleparts" if a downstream tool expects `Polygon`.
- **GeoJSON paste failures:** BOM characters, line breaks, or copy truncation cause silent failures when submitting to web tools. Minify to a single line via `json.load`/`json.dump` in Python before pasting.

## Project Structure Convention

```
data/raw/           # original downloads, untouched
data/processed/     # reprojected/clipped/derived rasters
                    # naming: <fire_name>_<layer>_<step>.tif
```

## Running the Downloader

```bash
pip install -r requirements.txt

# Download all tiles (single-threaded)
python download_urls.py AirplaneLakeLIDAR_DowloadList.txt --output_directories laz_files

# Parallel download (recommended)
python download_urls.py AirplaneLakeLIDAR_DowloadList.txt --output_directories laz_files --threads 4

# Filter by filename pattern
python download_urls.py AirplaneLakeLIDAR_DowloadList.txt --output_directories laz_files --match_regexp ".*10UFU4.*"

# Download from a live URL
python download_urls.py https://example.com/listing.html --output_directories laz_files --match_regexp ".*\.laz$"
```

## Code Architecture

**`download_urls.py`** is the main entry point. It accepts one or more input files (local paths or HTTP URLs) containing link lists, extracts all HTTPS URLs and `href` attributes via regex, and dispatches parallel downloads using `multiprocessing.Pool`. Each tile is downloaded via `curl` with automatic retry, written to a temp file first then moved to the final path — avoids leaving partial files on failure.

**`interrupt.py`** provides the `@handle_ctrl_c` decorator and `init_pool` initializer for clean Ctrl-C behavior in multiprocessing pools. Without this pattern, Ctrl-C leaves zombie worker processes. The decorator catches `KeyboardInterrupt` per worker and returns it as a value; the main process checks results for `KeyboardInterrupt` instances and exits.

## Data Notes

- LAZ files are large (hundreds of MB each) and are gitignored under `laz_files/`
- Tile filenames encode UTM grid coordinates (e.g., `10UFU4019` = zone 10U, grid FU, easting 40, northing 19)
- The downloader skips tiles that already exist with non-zero size — re-runs are safe
- Some tiles have `.copc.laz` variants (Cloud-Optimized Point Cloud) alongside standard `.laz`
