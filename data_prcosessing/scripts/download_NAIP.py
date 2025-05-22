import pandas as pd
import os
import requests
import planetary_computer as pc
from pystac_client import Client
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml

# ----------------------------------------------------------------------
# Load configuration from YAML
# ----------------------------------------------------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

download_config = config["download"]
csv_path = download_config["location_csv"]
output_root = download_config["naip_root"]
lat_col = download_config["lat_col"]
lon_col = download_config["lon_col"]
state_col = download_config["location_column"]
folder = download_config["folder"]
max_workers = download_config["max_workers"]

# ----------------------------------------------------------------------
# Load your points
# ----------------------------------------------------------------------
df = pd.read_csv(csv_path)

# ----------------------------------------------------------------------
# Output folder
# ----------------------------------------------------------------------
os.makedirs(output_root, exist_ok=True)

# ----------------------------------------------------------------------
# Connect to Planetary Computer STAC API
# ----------------------------------------------------------------------
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# ----------------------------------------------------------------------
# Initialize structures
# ----------------------------------------------------------------------
downloaded_images = {}  # key: asset_href, value: local filename
failed_points = []
point_image_mapping = []  # list of mappings: point -> image file
# ----------------------------------------------------------------------
# Define download function
# ----------------------------------------------------------------------
def download_naip(idx, lat, lon, state):
    try:
        search = catalog.search(
            collections=["naip"],
            intersects=Point(lon, lat),
            limit=1
        )

        items = list(search.get_items())
        if len(items) == 0:
            return {'fail': True, 'point': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'reason': 'No imagery found'}}

        item = items[0]
        asset_href = pc.sign(item.assets['image'].href)

        # Create state folder
        state_folder = os.path.join(output_root, state + '/images')
        os.makedirs(state_folder, exist_ok=True)

        # Generate filename based on asset id
        asset_id = item.id.replace("/", "_").replace(":", "_")  # Clean id for filename
        output_filename = os.path.join(state_folder, f"{asset_id}.tif")

        # Check if already downloaded
        if asset_href in downloaded_images:
            existing_file = downloaded_images[asset_href]
            return {'fail': False, 'mapping': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'image_file': existing_file}}

        # If file already exists on disk (by chance)
        if os.path.exists(output_filename):
            print(f"Already downloaded {output_filename}, skipping...")
            downloaded_images[asset_href] = output_filename
            return {'fail': False, 'mapping': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'image_file': output_filename}}

        # Download the image
        print(f"Downloading image for point {idx} ({state})...")
        r = requests.get(asset_href, stream=True)
        if r.status_code == 200:
            with open(output_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_images[asset_href] = output_filename
            print(f"Saved {output_filename}")
            return {'fail': False, 'mapping': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'image_file': output_filename}}
        else:
            return {'fail': True, 'point': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'reason': f'Download failed, status {r.status_code}'}}

    except Exception as e:
        return {'fail': True, 'point': {'point_index': idx, 'lat': lat, 'lon': lon, 'state': state, 'reason': f'Exception: {str(e)}'}}

# ----------------------------------------------------------------------
# Launch parallel downloads
# ----------------------------------------------------------------------
with ThreadPoolExecutor(max_workers=40) as executor:
    futures = []
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col] 
        state = row[state_col]
        futures.append(executor.submit(download_naip, idx, lat, lon, state))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading NAIP Images"):
        result = future.result()
        if result['fail']:
            failed_points.append(result['point'])
        else:
            point_image_mapping.append(result['mapping'])

# ----------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------
if failed_points:
    failed_df = pd.DataFrame(failed_points)
    state_folder = os.path.join(output_root, folder + '/downlaod_info')
    os.makedirs(state_folder, exist_ok=True)
    failed_df.to_csv(f'{state_folder}/failed_download_points_unique_parallel.csv', index=False)
    print(f"\n Saved {len(failed_points)} failed points to failed_download_points_unique_parallel.csv")
else:
    print("\n No failed points!")

# Save point to image mapping
mapping_df = pd.DataFrame(point_image_mapping)
state_folder = os.path.join(output_root, folder + '/downlaod_info')
os.makedirs(state_folder, exist_ok=True)
mapping_df.to_csv(f'{state_folder}/point_to_image_mapping.csv', index=False)
print("\n Saved point-to-image mapping to point_to_image_mapping.csv!")
