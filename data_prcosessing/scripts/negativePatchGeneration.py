import os
import rasterio
import pandas as pd
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box, Point
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def create_geo_dataframe(df, lat_col="latitude", lon_col="longitude", crs="EPSG:4326"):
    df['geometry'] = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

def extract_patch_safe(src, point, patch_size):
    x, y = point.x, point.y
    row, col = src.index(x, y)
    half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
    if (row - half_h < 0 or col - half_w < 0 or
        row + half_h > src.height or col + half_w > src.width):
        return None, None
    window = Window(col - half_w, row - half_h, patch_size[1], patch_size[0])
    transform = src.window_transform(window)
    patch = src.read(window=window)
    return patch, transform

def process_single_tif(args):
    tif_path, gdf_wgs84, patch_size, output_folder = args
    saved = 0
    tif_name = os.path.splitext(os.path.basename(tif_path))[0]

    try:
        with rasterio.open(tif_path) as src:
            bounds = box(*src.bounds)
            gdf_proj = gdf_wgs84.to_crs(src.crs)
            candidates = gdf_proj[gdf_proj.geometry.within(bounds)]

            if candidates.empty:
                return 0

            for idx, row in candidates.iterrows():
                out_path = os.path.join(output_folder, f"{tif_name}_{saved}_neg.tif")
                if os.path.exists(out_path):
                    continue  # Skip if already exists

                patch, transform = extract_patch_safe(src, row.geometry, patch_size)
                if patch is None:
                    continue

                meta = src.meta.copy()
                meta.update({
                    "height": patch.shape[1],
                    "width": patch.shape[2],
                    "transform": transform
                })

                with rasterio.open(out_path, "w", **meta) as dest:
                    dest.write(patch)
                saved += 1

    except Exception as e:
        print(f"Removing unreadable file: {tif_path} — {e}")
        try:
            os.remove(tif_path)
        except Exception as del_e:
            print(f"Could not delete {tif_path}: {del_e}")
    return saved

def process_points_parallel(df, tif_folder, patch_size=(224, 224), output_folder="patches", max_workers=None):
    os.makedirs(output_folder, exist_ok=True)
    gdf = create_geo_dataframe(df)
    tif_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(tif_folder)
        for f in files if f.endswith('.tif')
    ]

    print(f"Starting parallel extraction on {len(tif_files)} TIFFs with {multiprocessing.cpu_count()} workers...")

    args_list = [(tif_path, gdf, patch_size, output_folder) for tif_path in tif_files]

    total_saved = 0
    with ProcessPoolExecutor(max_workers=max_workers or multiprocessing.cpu_count()) as executor:
        for result in tqdm(executor.map(process_single_tif, args_list), total=len(args_list)):
            total_saved += result

    print(f"\nDone. Extracted {total_saved} patches to '{output_folder}'")

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("/project/biocomplexity/wyr6fx(Nibir)/Dump/RemoteSensing Dump/Soil-USA/LandMask/filtered_landuse_points_min2km_from_CAFO.csv")
    process_points_parallel(
        df,
        tif_folder="/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/scripsts/downloaded_naip_full_v2/NEG_SAMPLE",
        patch_size=(833, 833),
        output_folder="/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/negative_sample"
    )
