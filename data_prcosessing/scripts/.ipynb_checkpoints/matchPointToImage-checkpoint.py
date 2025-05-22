# -----------------------------------------------------------------------------
# This script matches geospatial points (from a GeoDataFrame `gdf_proj`) to 
# NAIP imagery tiles (.tif files).
#
# For each NAIP tile:
# - Open the GeoTIFF file using rasterio.
# - Get its bounding box (spatial extent).
# - Check if any of the points from `gdf_proj` fall within that bounding box.
# - If a point is contained, record the point's coordinates, the matching image filename, and folder.
#
# The final output is saved as an Excel file ("point_image_mapping.xlsx"), 
# listing which point is contained in which NAIP tile.
#
# Requirements:
# - `gdf_proj` must be a GeoDataFrame with points and CRS matching the NAIP imagery (likely EPSG:3857).
# - NAIP imagery (.tif) must be stored under `naip_root` path, potentially in subfolders.
#
# -----------------------------------------------------------------------------


import os
import yaml
import rasterio
import pandas as pd
from shapely.geometry import box
import geopandas as gpd


import os
import yaml
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box, Point
from utils import *
import sys



def match_points_to_images(gdf_proj, config):
    mp_config = config['match_point']
    results = []

    states = mp_config['states_of_interest'] or gdf_proj[mp_config['location_column']].unique().tolist()
    print(f"States of interest: {states}")

    for state_folder in os.listdir(config['naip_root']):
        if state_folder in states:
            state_path = os.path.join(config['naip_root'], state_folder)
            print(f"Processing state: {state_folder}")

            for root, dirs, files in os.walk(state_path):
                for file in files:
                    if file.endswith(".tif"):
                        tif_path = os.path.join(root, file)
                        try:
                            with rasterio.open(tif_path) as src:
                                bounds = src.bounds
                                image_box = box(*bounds)

                                state_points = gdf_proj[gdf_proj[mp_config['location_column']] == state_folder]
                                state_points_proj = state_points.to_crs(src.crs)

                                for idx, row in state_points_proj.iterrows():
                                    if image_box.contains(row.geometry):
                                        point = row.geometry
                                        x, y = point.x, point.y
                                        results.append({
                                            'CAFO_UNIQUE_ID': row[mp_config['cafo_unique_id']],
                                            'x': row[mp_config['lon_col']],
                                            'y': row[mp_config['lat_col']],
                                            'geo_x': x,
                                            'geo_y': y,
                                            'CAFO_TYPE': row[mp_config['cafo_type']],
                                            'STATE': row[mp_config['location_column']],
                                            'image_file': file,
                                            'image_folder': os.path.basename(root)
                                        })
                        except Exception as e:
                            print(f"Could not read {tif_path}: {e}")
    return results


def save_results(results, gdf_proj, config):
    mp_config = config['match_point']
    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(config['point_image_mapping_csv']), exist_ok=True)
    output_df.to_csv(config['point_image_mapping_csv'], index=False)
    print(f"Saved matched results to {config['point_image_mapping_csv']}")

    unmatched_ids = set(gdf_proj[mp_config['cafo_unique_id']]) - set(output_df['CAFO_UNIQUE_ID'])
    unmatched_df = pd.DataFrame({'CAFO_UNIQUE_ID': list(unmatched_ids)})
    os.makedirs(os.path.dirname(mp_config['output_unmatched_csv']), exist_ok=True)
    unmatched_df.to_csv(mp_config['output_unmatched_csv'], index=False)
    print(f"Saved unmatched IDs to {mp_config['output_unmatched_csv']}")


def main(yaml_path="config.yaml"):
    config = load_config(yaml_path)
    config = resolve_placeholders(config)
    mp_config = config['match_point']

    input_csv = config.get("location_csv")
    if not input_csv:
        raise ValueError("`location_csv` must be defined in the YAML under general.")

    print(f"Loading input CSV from {input_csv}")
    df = pd.read_csv(input_csv)

    # Convert to GeoDataFrame
    lon_col = mp_config['lon_col']
    lat_col = mp_config['lat_col']
    df = df.dropna(subset=[lon_col, lat_col])
    df['geometry'] = df.apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    gdf_proj = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    results = match_points_to_images(gdf_proj, config)
    save_results(results, gdf_proj, config)


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_path)