import os
import torch
import rasterio
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torchcam.methods import GradCAM
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from shapely.geometry import Polygon, Point
from skimage import measure
import geopandas as gpd
import sys
import yaml
from tqdm import tqdm
sys.path.append('/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/')
from model.classifierModel import CAFOClassifier
from utils import *

# ========== LOAD CONFIGURATION ==========
def get_config(yaml_path="config.yaml"):
    full_config = load_config(yaml_path)
    resolved_config = resolve_placeholders(full_config)
    resolved_config["refinement"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return resolved_config["refinement"]


# ========== MODEL AND TRANSFORM SETUP ==========
def load_model(CONFIG):
    model = CAFOClassifier.load_from_checkpoint(CONFIG['model_path'])
    model.to(CONFIG['device']).eval()
    return model

def get_transform(CONFIG):
    return transforms.Compose([
        transforms.Resize(CONFIG['input_size']),
        transforms.ToTensor(),
    ])


# ========== IMAGE AND CAM PROCESSING ==========
def read_image_with_metadata(image_path):
    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3]).astype(np.float32) / 255.0
        transform = src.transform
        crs = src.crs
        meta = src.meta
    return img, transform, crs, meta

def run_gradcam(model, extractor, img_tensor):
    with torch.enable_grad():
        img_tensor.requires_grad_()
        output = model(img_tensor)
        pred_class = output.argmax().item()
        activation_map = extractor(pred_class, output)[0]
    return activation_map, pred_class

def extract_cam_center(cam_map, transform, image_meta, img_shape):
    resized = F.interpolate(cam_map.unsqueeze(1), size=img_shape, mode='bilinear', align_corners=False)
    heatmap = resized.squeeze().cpu().numpy()
    center_y, center_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    original_height = image_meta['height']
    original_width = image_meta['width']

    scaled_x = center_x * (original_width / img_shape[1])
    scaled_y = center_y * (original_height / img_shape[0])

    projected_x, projected_y = transform * (scaled_x, scaled_y)
    return heatmap, center_x, center_y, projected_x, projected_y


# ========== POLYGON HANDLING ==========
def cam_to_polygons(heatmap, threshold):
    binary_mask = (heatmap > threshold * heatmap.max()).astype(np.uint8)
    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []

    for contour in contours:
        if len(contour) >= 3:
            poly = Polygon([(x, y) for y, x in contour])
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
    return polygons

def reproject_polygon(polygon, transform, heatmap_shape, image_meta):
    scale_x = image_meta['width'] / heatmap_shape[1]
    scale_y = image_meta['height'] / heatmap_shape[0]
    def scale_coords(x, y):
        return transform * (x * scale_x, y * scale_y)
    return Polygon([scale_coords(x, y) for x, y in polygon.exterior.coords])

def save_polygon_geojson(polygons, crs, output_path):
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(output_path, driver="GeoJSON")


# ========== MAIN PROCESSING ==========
def process_image(model, extractor, transform_fn, image_path, image_to_metadata, CONFIG):
    fname = os.path.basename(image_path)
    if fname not in image_to_metadata:
        print(f"⚠️ Skipping {fname} — not found in metadata CSV.")
        return None

    meta = image_to_metadata[fname]
    unique_id = meta.get("CAFO_UNIQUE_ID", "NA")
    patch_id = meta.get("patch_id", "NA")
    weak_x = meta.get("main_x")
    weak_y = meta.get("main_y")
    geo_x = meta.get("geo_x")
    geo_y = meta.get("geo_y")
    state = meta.get("state")
    cafo_type = meta.get("cafo_type")
    crs = meta.get("crs")

    image_tensor, geo_transform, crs, raster_meta = read_image_with_metadata(image_path)

    img_np = np.transpose(image_tensor, (1, 2, 0))
    img_shape = CONFIG['cam_output_size']
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    input_tensor = transform_fn(img_pil).unsqueeze(0).to(CONFIG['device'])

    cam_map, pred_class = run_gradcam(model, extractor, input_tensor)
    if pred_class != 1:
        return {
            'CAFO_UNIQUE_ID': unique_id,
            'patch_file': fname,
            'patch_id': patch_id,
            'main_x': weak_x,
            'main_y': weak_y,
            'geo_x': geo_x,
            'geo_y': geo_y,
            'state': state,
            'cafo_type': cafo_type,
            'cam_center_pixel_x': None,
            'cam_center_pixel_y': None,
            'projected_x': None,
            'projected_y': None,
            'predicted_class': pred_class,
            'polygon_file': None,
            'state': state,
            'cafo_type': cafo_type,
            'crs': crs
        }

    heatmap, center_x, center_y, proj_x, proj_y = extract_cam_center(cam_map, geo_transform, raster_meta, img_shape)

    polygons = cam_to_polygons(heatmap, CONFIG['cam_threshold'])
    reprojected_polys = [reproject_polygon(p, geo_transform, heatmap.shape, raster_meta) for p in polygons]

    os.makedirs(CONFIG['polygon_dir'], exist_ok=True)
    poly_filename = os.path.splitext(fname)[0] + "_cam.geojson"
    poly_path = os.path.join(CONFIG['polygon_dir'], poly_filename)
    save_polygon_geojson(reprojected_polys, crs, poly_path)

    return {
        'CAFO_UNIQUE_ID': unique_id,
        'patch_id': patch_id,
        'patch_file': fname,
        'main_x': weak_x,
        'main_y': weak_y,
        'geo_x': geo_x,
        'geo_y': geo_y,
        'cam_center_pixel_x': int(center_x),
        'cam_center_pixel_y': int(center_y),
        'projected_x': proj_x,
        'projected_y': proj_y,
        'predicted_class': pred_class,
        'polygon_file': poly_filename,
        'state': state,
        'cafo_type': cafo_type,
        'crs': crs
    }


# ========== MAIN SCRIPT ==========
def main(yaml_path="config.yaml"):
    # print('Here!!!!')
    CONFIG = get_config(yaml_path)

    weak_df = pd.read_csv(CONFIG['multi_patch_metadata_csv'])
    image_to_metadata = weak_df.set_index("patch_file").to_dict(orient="index")
    
    
    print('*******************Loading GradCAM***************************')
    model = load_model(CONFIG)
    extractor = GradCAM(model.model, target_layer=CONFIG['target_layer'])
    transform_fn = get_transform(CONFIG)
    print('*******************Loading Completed*************************')
    print('*******************Running Refining Algorithm****************')
    results = []
    for fname in tqdm(os.listdir(CONFIG['image_dir']), desc="Processing images"):
        if fname.lower().endswith('.tif'):
            image_path = os.path.join(CONFIG['image_dir'], fname)
            try:
                result = process_image(model, extractor, transform_fn, image_path, image_to_metadata, CONFIG)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

    cam_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CONFIG['weak_csv']), exist_ok=True)
    cam_df.to_csv(CONFIG['weak_csv'], index=False)
    print(f"Saved CAM results to {CONFIG['weak_csv']}")


if __name__ == "__main__":
    # print('Running Refining Coords!!')
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(yaml_path)