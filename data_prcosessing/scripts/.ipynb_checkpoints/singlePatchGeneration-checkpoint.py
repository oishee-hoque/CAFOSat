import os
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading
from rasterio.windows import Window
import sys
from utils import *
import yaml

    
def process_image(group_key, group_df, config, patch_id_counter, lock):
    results = []
    image_file, image_folder = group_key
    image_path = os.path.join(config['naip_root'], image_folder, image_file)

    try:
        with rasterio.open(image_path) as src:
            pixel_x, _ = src.res
            buffer_meters = config['crop_buffer'] / 2 if pixel_x <= 0.3 else config['crop_buffer']
            crs = src.crs

            for idx, row in group_df.iterrows():
                try:
                    x_center, y_center = row['geo_x'], row['geo_y']
                    x, y = row['x'], row['y']
                    unique_id = row['CAFO_UNIQUE_ID']
                    cafo_type = row['CAFO_TYPE']
                    state = row['STATE']

                    px, py = ~src.transform * (x_center, y_center)
                    col_off = int(px) - config['patch_size'] // 2
                    row_off = int(py) - config['patch_size'] // 2
                    col_off = max(0, col_off)
                    row_off = max(0, row_off)

                    window = Window(col_off, row_off, 833, 833)
                    crop = src.read(window=window)
                    out_transform = src.window_transform(window)

                    if crop.shape[1] < 100 or crop.shape[2] < 100 or np.all(crop == 0):
                        continue

                    with lock:
                        patch_id = patch_id_counter['value']
                        patch_id_counter['value'] += 1

                    patch_filename = f"crop_{idx+1}_patch_{patch_id}_{cafo_type}_{state}.tif"
                    patch_path = os.path.join(config['patch_output_dir'], patch_filename)

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": crop.shape[1],
                        "width": crop.shape[2],
                        "count": 4,
                        "transform": out_transform,
                    })

                    with rasterio.open(patch_path, "w", **out_meta) as dst:
                        dst.write(crop)

                    minx, miny = out_transform * (0, crop.shape[0])
                    maxx, maxy = out_transform * (crop.shape[1], 0)

                    results.append({
                        'CAFO_UNIQUE_ID': unique_id,
                        'patch_id': patch_id,
                        'source_point': idx,
                        'patch_file': patch_filename,
                        'image_folder': row['image_folder'],
                        'image_file': image_file,
                        'main_x': x,
                        'main_y': y,
                        'geo_x': x_center,
                        'geo_y': y_center,
                        'cafo_type': cafo_type,
                        'state': state,
                        'crs': crs,
                        'res': pixel_x
                    })

                except Exception as e:
                    print(f"Error processing point in {image_path}: {e}")

    except Exception as e:
        print(f"Error opening {image_path}: {e}")

    return results

def extract_patches_parallel(grouped, config, patch_id_counter, lock):
    all_results = []
    with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
        futures = [executor.submit(process_image, group_key, group_df, config, patch_id_counter, lock)
                   for group_key, group_df in grouped]
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results

def main(yaml_path="config.yaml"):
    full_config = load_config(yaml_path)
    full_config = resolve_placeholders(full_config)
    config = full_config['single_patch'] 
    config['naip_root'] = full_config['naip_root']
    os.makedirs(config['patch_output_dir'], exist_ok=True)
    df_cafo = pd.read_csv(full_config['point_image_mapping_csv'])
    grouped = df_cafo.groupby(['image_file', 'image_folder'])

    patch_id_counter = {'value': 1}
    lock = threading.Lock()

    print("Starting patch extraction...")
    results = extract_patches_parallel(grouped, config, patch_id_counter, lock)

    metadata_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(config['metadata_csv']), exist_ok=True)
    metadata_df.to_csv(config['metadata_csv'], index=False)
    print(f"Done! Total patches created: {len(metadata_df)}")
    print(f"Metadata saved to: {config['metadata_csv']}")

if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_path)