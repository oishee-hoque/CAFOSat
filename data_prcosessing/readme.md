# 🧪 Pipeline Setup Guide

## 1️⃣ Environment Setup

To ensure compatibility and reproducibility, please begin by setting up your environment using the provided `requirements.txt` file.

### 🔧 Step-by-Step

#### ➤ Option A: Using `virtualenv` or `venv`

```bash
# Create a virtual environment
python -m venv cafosat_env
source cafosat_env/bin/activate  # On Windows use: cafosat_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

#### ➤ Option B: Using `conda`

```bash
# Create and activate a conda environment
conda create -n cafosat_env python=3.10
conda activate cafosat_env

# Install pip dependencies
pip install -r requirements.txt
```

## 🌐 Step 2: Download NAIP Imagery (Optional)

If you **already have NAIP imagery downloaded**, you can skip this step and proceed to patch generation.  
Otherwise, use the `download_NAIP.py` script to fetch imagery using geolocated points via Microsoft's Planetary Computer.

---

### 📝 Required Input

Prepare a CSV file containing point locations. The file should include at least the following columns (column names can be customized via the config):

```csv
latitude,longitude,state
38.8977,-77.0365,DC
34.0522,-118.2437,CA
...
```
🔍 See example CSVs in: [`/datas/input_csvs`](../datas/input_csvs)

### 🗂 Output Directory Structure: `naip_downloads/`

After running the `download_NAIP.py` script, the NAIP imagery and metadata will be saved in the `naip_downloads/` folder.

```text
naip_downloads/
├── DC/
│   └── images/
│       └── <image_id>.tif
├── CA/
│   └── images/
│       └── <image_id>.tif
└── example_run/
    └── download_info/
        ├── point_to_image_mapping.csv
        └── failed_download_points_unique_parallel.csv
```

## ⚙️ Step 3: Pipeline Execution Options

You can run the entire CAFOSat pipeline using a single wrapper script **or** execute each step individually depending on your needs.

---

### ✅ Option 1: Run the Full Pipeline

Use the `run_pipeline.py` script to execute all processing stages in sequence, based on which steps are enabled in your `config.yaml`.

```bash
python run_pipeline.py config.yaml
```

#### 🔧 Required Configuration

Make sure your `config.yaml` contains a `run_flags` section to control which steps of the pipeline to execute.

```yaml
run_flags:
  match_point_to_image: true
  multi_patch_generate: true
  single_patch_generate: true
  refine_coords: true
  cluster: true
  single_patch_filtered: true
```
#### 📝 Required Columns in the CSV

These columns must match the keys specified under `match_point:` in your `config.yaml` to begin running the pipeline.

| Column Name (in CSV) | Description                                                    | Example            |
|----------------------|----------------------------------------------------------------|--------------------|
| `CAFO_UNIQUE_ID`     | A unique identifier for each CAFO or location                  | `US123456`         |
| `x`                  | Longitude of the point (EPSG:4326)                             | `-93.6111`         |
| `y`                  | Latitude of the point (EPSG:4326)                              | `42.0322`          |
| `STATE`              | location code                                                  | `IA`               |
| `CAFO_TYPE`          | Category/type of the CAFO (e.g., Swine, Poultry, Dairy, etc.)  | `Swine`            |

⚠️ **Note:**  
The column names in your CSV must exactly match the values you provide in the `config.yaml` under `match_point`. These are not optional.

---

### 🛠 Option 2: Run Each Script Manually

You can also execute each script individually for greater control, debugging, or selective step execution. Below is a breakdown of what each script does and its inputs/outputs.

---

#### `matchPointToImage.py`

🔍 Matches geospatial CAFO points to downloaded NAIP imagery tiles based on spatial overlap.

- 📥 **Input**: Geolocated point CSV
- 📤 **Output**: `point_image_mapping.csv`, `unmatched_points.csv`

---

#### `multiplePatchGeneration.py`

📦 Generates multiple image patches per matched NAIP tile, typically for data augmentation or tile sampling.

- 📥 **Input**: `point_image_mapping.csv`
- 📤 **Output**: multiple patches around given geo location - 'multi_patch_metadata_{state}.csv', 'point_multi_patches/{state}/<patch_id>.tif'

---

#### `singlePatchGeneration.py`

🎯 Creates a single, center-aligned patch (e.g., 833×833 pixels) for each matched point.

- 📥 **Input**: `point_image_mapping.csv`
- 📤 **Output**: One `.tif` patch per given geo location: 'point_single_patches/{state}/<patch_id>.tif'

---

#### `refiningCoords.py`

🧭 Refines coordinates for CAFO infrastructure detection using auxiliary logic (e.g., masks, object proposals).

- 📥 **Input**: trained cafo classifier (you can download our trained model here and use it), multi_patch_metadata_csv, multi_patces
- 📤 **Output**: Refined coordinates and updated metadata: 'refined_coords_csvs/cafo_cam_projected_centers_{state}.csv', 'cam_polygons/{state}'

---

#### `clustering.py`

🔗 Clusters detected infrastructure elements (e.g., barns, lagoons) into coherent groups representing full CAFO facilities.

- 📥 **Input**:  Refined coordinates and updated metadata, 
- 📤 **Output**: Clustered `GeoDataFrame` and associated metadata: cam_polygons/filtered_cam_polygons_{state}.geojson, cam_polygons/filtered_cam_polygons_{state}.csv

---

#### `singlePatchGenerationfiltered.py`

🧹 Generates a **filtered** version of patches using refined coordinates or clustered results.

- 📥 **Input**: Refined or clustered coordinates
- 📤 **Output**: Filtered final patch set: '/patch_metadata/single_patch_metadata_{state}_filtered.csv', '/point_single_patches/{state}_filtered'

---

#### 🧾 Summary

| Script                           | Description                               | Depends On                        |
|----------------------------------|-------------------------------------------|-----------------------------------|
| `matchPointToImage.py`           | Match points to imagery tiles             | Point CSV, NAIP `.tif` files      |
| `multiplePatchGeneration.py`     | Generate multiple patches per image       | `point_image_mapping.csv`         |
| `singlePatchGeneration.py`       | Create single center-aligned patches      | `point_image_mapping.csv`         |
| `refiningCoords.py`              | Refine patch center coordinates           | Patches, model output             |
| `clustering.py`                  | Cluster detected infrastructure           | Refined coordinates               |
| `singlePatchGenerationfiltered.py` | Generate final filtered patches         | Clustering/refinement outputs     |

---

📝 You can run any script independently by passing your `config.yaml`:

```bash
python script_name.py config.yaml
```


