import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
from rasterio.windows import from_bounds
import rasterio
from rasterio.transform import Affine
import numpy as np

def crop_and_patchify(row_data, config):
    row, idx, patch_id_start = row_data
    patch_metadata = []
    patch_id = patch_id_start

    try:
        image_path = os.path.join(config['naip_root'], row['image_folder'], row['image_file'])

        with rasterio.open(image_path) as src:
            pixel_x, _ = src.res
            buffer_meters = config['crop_buffer'] / 2 if pixel_x <= 0.3 else config['crop_buffer']
            window = from_bounds(
                row['geo_x'] - buffer_meters, row['geo_y'] - buffer_meters,
                row['geo_x'] + buffer_meters, row['geo_y'] + buffer_meters,
                src.transform
            )

            out_transform = src.window_transform(window)
            crop = src.read(config['bands'], window=window)
            crop_rows, crop_cols = crop.shape[1], crop.shape[2]

            if crop_rows < config['patch_size'] or crop_cols < config['patch_size']:
                return []

            # Center patch
            center_off_row = (crop_rows - config['patch_size']) // 2
            center_off_col = (crop_cols - config['patch_size']) // 2
            center_patch = crop[:, center_off_row:center_off_row + config['patch_size'],
                                     center_off_col:center_off_col + config['patch_size']]

            if center_patch.shape[1] == config['patch_size'] and center_patch.shape[2] == config['patch_size']:
                transform = out_transform * Affine.translation(center_off_col, center_off_row)
                meta = create_patch_metadata(patch_id, idx, 'center', center_patch, transform, src, config, row,
                                             f"center_{row['CAFO_TYPE']}_{row['STATE']}")
                patch_metadata.append(meta)
                patch_id += 1

            # Tiled patches
            for r in range(0, crop_rows, config['patch_size']):
                for c in range(0, crop_cols, config['patch_size']):
                    patch = crop[:, r:r + config['patch_size'], c:c + config['patch_size']]
                    if patch.shape[1] == 0 or patch.shape[2] == 0 or np.all(patch == 0):
                        continue

                    transform = out_transform * Affine.translation(c, r)
                    meta = create_patch_metadata(patch_id, idx, 'tile', patch, transform, src, config, row,
                                                 f"{row['CAFO_TYPE']}_{row['STATE']}")
                    patch_metadata.append(meta)
                    patch_id += 1

        return patch_metadata

    except Exception as e:
        print(f"Error processing point {idx}: {e}")
        return []


def main(yaml_path="config.yaml"):
    full_config = load_config(yaml_path)
    full_config = resolve_placeholders(full_config)
    config = full_config['multi_patch']
    config['naip_root'] = full_config['naip_root']

    os.makedirs(config['output_patch_dir'], exist_ok=True)
    df = pd.read_csv(full_config['point_image_mapping_csv'])
    tasks = [(row, idx, idx * 10000 + 1) for idx, row in df.iterrows()]
    all_metadata = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(crop_and_patchify, task, config) for task in tasks]
        for future in as_completed(futures):
            all_metadata.extend(future.result())
            
            
    os.makedirs(os.path.dirname(config['multi_patch_metadata_csv']), exist_ok=True)
    pd.DataFrame(all_metadata).to_csv(config['multi_patch_metadata_csv'], index=False)
    print(f"Done. Saved {len(all_metadata)} patches to {config['multi_patch_metadata_csv']}")


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_path)
