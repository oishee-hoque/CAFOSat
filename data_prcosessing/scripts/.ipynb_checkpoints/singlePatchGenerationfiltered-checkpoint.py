import os
import sys
import yaml
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils import *
from rasterio.errors import RasterioIOError


# ========== CORE FUNCTION ========== #
def process_image(group_key, group_df, config, id_to_patchfile, id_to_patchid, image_folder, lock):
    results = []
    image_file, image_folder = group_key
    image_path = os.path.join(config['naip_root'], image_folder, image_file)

    try:
        with rasterio.open(image_path) as src:
            pixel_x, _ = src.res
            crs = src.crs

            for idx, row in group_df.iterrows():
                try:
                    x_center, y_center = row['refined_x'], row['refined_y']
                    unique_id = row['CAFO_UNIQUE_ID']
                    patch_filename = id_to_patchfile.get(unique_id)
                    patch_id = id_to_patchid.get(unique_id)
                    if not patch_filename:
                        print(f"No patch filename found for unique_id: {unique_id}")
                        continue


                    px, py = ~src.transform * (x_center, y_center)
                    col_off = int(px) - config['patch_size'] // 2
                    row_off = int(py) - config['patch_size'] // 2
                    col_off = max(0, col_off)
                    row_off = max(0, row_off)

                    window = Window(col_off, row_off, config['patch_size'], config['patch_size'])
                    patch = src.read([1, 2, 3,4], window=window)
                    patch_transform = src.window_transform(window)

                    if patch.shape[1] < 100 or patch.shape[2] < 100 or np.all(patch == 0):
                        continue

                    patch_path = os.path.join(config['patch_output_dir'], patch_filename)
                    os.makedirs(config['patch_output_dir'], exist_ok=True)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": patch.shape[1],
                        "width": patch.shape[2],
                        "count": 4,
                        "transform": patch_transform,
                        "crs": crs
                    })

                    with rasterio.open(patch_path, "w", **out_meta) as dst:
                        dst.write(patch)

                    minx, miny = patch_transform * (0, patch.shape[1])
                    maxx, maxy = patch_transform * (patch.shape[2], 0)

                    results.append({
                        'CAFO_UNIQUE_ID': unique_id,
                        'patch_id': patch_id,
                        'source_point': idx,
                        'patch_file': patch_filename,
                        'image_folder': image_folder,
                        'image_file': image_file,
                        'main_x': row['main_x'],
                        'main_y': row['main_y'],
                        'geo_x': row['geo_x'],
                        'geo_y': row['geo_y'],
                        'refined_x': x_center,
                        'refined_y': y_center,
                        'cafo_type': row['cafo_type'],
                        'state': row['state'],
                        'crs': crs,
                        'res': pixel_x
                    })

                except Exception as e:
                    print(f"Error processing row in {image_path}: {e}")

    except RasterioIOError as e:
        print(f"Error opening image {image_path}: {e}")

    return results

# ========== MAIN FUNCTION ========== #
def main(yaml_path="config.yaml"):
    full_config = load_config(yaml_path)
    full_config = resolve_placeholders(full_config)
    config = full_config['filtered_single_patch']
    config['naip_root'] = full_config['naip_root']

    os.makedirs(config['patch_output_dir'], exist_ok=True)
    df_cafo = pd.read_csv(config['mapping_file'])
    image_folder = df_cafo['image_folder']
    grouped = df_cafo.groupby(['image_file', 'image_folder'])

    main_meta_df = pd.read_csv(config['main_metadata_file'])
    id_to_patchfile = dict(zip(main_meta_df['CAFO_UNIQUE_ID'], main_meta_df['patch_file']))
    id_to_patchid = dict(zip(main_meta_df['CAFO_UNIQUE_ID'], main_meta_df['patch_id']))


    lock = threading.Lock()

    results = []
    with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
        futures = [executor.submit(process_image, key, group, config, id_to_patchfile, id_to_patchid, image_folder, lock)
                   for key, group in grouped]
        for future in as_completed(futures):
            results.extend(future.result())

    os.makedirs(os.path.dirname(config['metadata_output_file']), exist_ok=True)
    pd.DataFrame(results).to_csv(config['metadata_output_file'], index=False)
    print(f"Done. Saved {len(results)} patches to {config['metadata_output_file']}")

# ========== ENTRY POINT ========== #
if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_path)
