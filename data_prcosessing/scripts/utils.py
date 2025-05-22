import yaml
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

##----------------------- CONFIG ---------------------------------------------------

def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def resolve_placeholders(config):
    top_vars = {k: v for k, v in config.items() if not isinstance(v, dict)}

    def resolve(obj):
        if isinstance(obj, str):
            return obj.format(**top_vars)
        elif isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(config)



##-----------------------multiPatch--------------------------------------------------

def create_patch_metadata(patch_id, idx, patch_type, patch, patch_transform, src, config, row, file_suffix):
    minx, miny = patch_transform * (0, config['patch_size'])
    maxx, maxy = patch_transform * (config['patch_size'], 0)
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

    patch_filename = f"crop_{idx+1}_patch_{patch_id}_{file_suffix}.tif"
    patch_path = os.path.join(config['output_patch_dir'], patch_filename)

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": patch.shape[1],
        "width": patch.shape[2],
        "count": len(config['bands']),
        "transform": patch_transform,
        "crs": src.crs
    })

    with rasterio.open(patch_path, "w", **out_meta) as dst:
        dst.write(patch)

    return {
        'CAFO_UNIQUE_ID': row['CAFO_UNIQUE_ID'],
        'patch_id': patch_id,
        'source_point': idx,
        'patch_file': patch_filename,
        'image_folder': row['image_folder'],
        'image_file': row['image_file'],
        'main_x': row['x'],
        'main_y': row['y'],
        'geo_x': row['geo_x'],
        'geo_y': row['geo_y'],
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'patch_center_x': center_x,
        'patch_center_y': center_y,
        'cafo_type': row['CAFO_TYPE'],
        'state': row['STATE'],
        'crs': str(src.crs),
        'type': patch_type
    }

