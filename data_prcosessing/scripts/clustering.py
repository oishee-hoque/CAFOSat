import os
import glob
import sys
import yaml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
import numpy as np
from utils import *  # Adjust path if needed
from shapely.ops import unary_union

def main(yaml_path="config.yaml"):
    # Load and resolve the full config
    full_config = resolve_placeholders(load_config(yaml_path))
    config = full_config['clustering']

    POLYGON_DIR = config['polygon_dir']
    WEAK_CSV = config['weak_csv']
    OUTPUT_GEOJSON = config['output_geojson']
    OUTPUT_CSV = config['output_csv']
    MERGE_EPS = config['merge_eps']
    CRS = full_config.get('crs', "EPSG:26918")  # fallback if not specified

    # ====== LOAD WEAK COORDINATES ======
    weak_df = pd.read_csv(WEAK_CSV)
    weak_df = weak_df.dropna(subset=['projected_x', 'projected_y'])
    weak_df['geometry'] = weak_df.apply(lambda row: Point(row['geo_x'], row['geo_y']), axis=1)
    weak_gdf = gpd.GeoDataFrame(weak_df, geometry='geometry', crs=CRS)
    # weak_gdf = weak_gdf.rename(columns={'image': 'patch'})

    # Load mapping file dynamically
    mapping_csv = full_config['point_image_mapping_csv']
    mapping_df = pd.read_csv(mapping_csv)
    weak_gdf = weak_gdf.merge(mapping_df[['CAFO_UNIQUE_ID', 'image_file', 'image_folder']], left_on='CAFO_UNIQUE_ID', right_on='CAFO_UNIQUE_ID', how='left')

    # ====== LOAD ALL CAM POLYGONS ======
    records = []
    for file in glob.glob(os.path.join(POLYGON_DIR, "*.geojson")):
        image_id = os.path.basename(file).replace("_cam.geojson", "")
        image_file = image_id + ".tif"
        gdf = gpd.read_file(file)

        match = weak_gdf[weak_gdf["patch_file"] == image_file]
        if match.empty:
            print(f"No match found for {image_file}")
            continue

        unique_id = match.iloc[0]["CAFO_UNIQUE_ID"]

        if gdf.empty or gdf.geometry.isnull().all():
            print(f"Empty or invalid geometry in {file}")
            continue

        for geom in gdf.geometry:
            if geom.is_valid and not geom.is_empty:
                records.append({
                    "geometry": geom,
                    "centroid": geom.centroid,
                    "CAFO_UNIQUE_ID": unique_id,
                    "patch_file": image_file
                })

    if not records:
        print("No valid geometries found to create GeoDataFrame.")
        return

    poly_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=CRS)


    # ====== PROCESS EACH UNIQUE_ID ======
    filtered_polys = []
    filtered_points = []

    for uid in poly_gdf['CAFO_UNIQUE_ID'].unique():
        group = poly_gdf[poly_gdf['CAFO_UNIQUE_ID'] == uid].reset_index(drop=True)

        if len(group) == 1:
            chosen = group
        else:
            used = set()
            largest_cluster = []
            largest_cluster_size = 0

            for i in range(len(group)):
                if i in used:
                    continue

                base_poly = group.loc[i, 'geometry']
                cluster_indices = [i]

                for j in range(len(group)):
                    if i == j or j in used:
                        continue

                    compare_poly = group.loc[j, 'geometry']
                    if base_poly.intersects(compare_poly) or base_poly.distance(compare_poly) < MERGE_EPS:
                        cluster_indices.append(j)

                # Update largest cluster if this one is bigger
                if len(cluster_indices) > largest_cluster_size:
                    largest_cluster_size = len(cluster_indices)
                    largest_cluster = cluster_indices

            used.update(largest_cluster)
            chosen = group.loc[largest_cluster]

        # Merge polygons and compute refined centroid
        merged_poly = unary_union(chosen.geometry)
        refined_centroid = merged_poly.centroid

        weak_row = weak_gdf[weak_gdf['CAFO_UNIQUE_ID'] == uid].iloc[0]

        filtered_polys.append({
            "geometry": merged_poly,
            "CAFO_UNIQUE_ID": uid,
            "refined_centroid_x": refined_centroid.x,
            "refined_centroid_y": refined_centroid.y,
            "weak_x": weak_row.geometry.x,
            "weak_y": weak_row.geometry.y,
            "image": weak_row['image_file'],
            "image_folder": full_config['state']
        })

        filtered_points.append({
            "CAFO_UNIQUE_ID": uid,
            "image_file": weak_row['image_file'],
            "image_folder": weak_row['image_folder'],
            "patch_id" : weak_row['patch_id'],
            "patch_file": weak_row['patch_file'],
            "main_x": weak_row['main_x'],
            "main_y": weak_row['main_y'],
            "geo_x": weak_row.geometry.x,
            "geo_y": weak_row.geometry.y,
            "refined_x": refined_centroid.x,
            "refined_y": refined_centroid.y,
            "distance": refined_centroid.distance(weak_row.geometry),
            "cafo_type": weak_row['cafo_type'],
            "state": weak_row['state'],
            "loc_crs": CRS,
            "image_crs": weak_row['crs']
            
        })
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    # Save output
    refined_gdf = gpd.GeoDataFrame(filtered_polys, geometry='geometry', crs=CRS)
    refined_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    pd.DataFrame(filtered_points).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(filtered_points)} filtered centroids to {OUTPUT_CSV}")
    print(f"Saved filtered polygons to {OUTPUT_GEOJSON}")


if __name__ == "__main__":
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_file)
