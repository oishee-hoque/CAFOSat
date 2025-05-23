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
🔍 See example CSVs in: [`/data/input_csvs`](../data/input_csvs)

## 🗂 Output Directory Structure: `naip_downloads/`

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

