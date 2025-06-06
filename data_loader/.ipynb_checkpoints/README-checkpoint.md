# 📦 CAFOSat Data Loader

This [module](https://github.com/oishee-hoque/CAFOSat/tree/main/data_loader/dataloader_v1.ipynb) provides tools and examples to download, load, and explore the [CAFOSat dataset](https://huggingface.co/datasets/oishee3003/CAFOSat) — a high-resolution remote sensing dataset for identifying Confined Animal Feeding Operations (CAFOs) across the United States.

---

## 📥 Download and Load CAFOSat Data

You can stream the dataset directly from Hugging Face or download `.tar.gz` files locally for permanent use.

#### 🛠 Requirements

```bash
pip install datasets huggingface_hub pillow
```

---

### 📦 Option 1: Download the Full Dataset with Git (Recommended for Local Use)

To download the entire CAFOSat dataset — including all `.tar.gz`, `.csv`, and metadata files — use `git lfs`:

```bash
# Install Git LFS (only once per machine)
git lfs install

# Clone the full dataset repository
git clone https://huggingface.co/datasets/oishee3003/CAFOSat

# Optionally rename the folder
mv CAFOSat CAFOSat_data
```

### 🔗 Option 2: Load the Full Dataset (Streaming or Cached)

```python
from datasets import load_dataset

# Load the full CAFOSat dataset
ds = load_dataset("oishee3003/CAFOSat")
```

---

```python
from huggingface_hub import hf_hub_download

data_path = hf_hub_download(
    repo_id="oishee3003/CAFOSat",
    filename="AL_filtered.tar.gz",       # 🔸 Required: specify file name here
    repo_type="dataset",
    local_dir="CAFOSat_data"             # Optional: where to save
)

print(f"Dataset downloaded to: {data_path}")
```
---

```python
from datasets import load_dataset

# Load a locally downloaded .tar.gz file
ds = load_dataset("oishee3003/CAFOSat", data_files="AR_filtered.tar.gz")
```

---

## 🖼️ Explore a Sample Image

```python
from PIL import Image
import io

# View one example
for row in ds['train']:
    print(row)
    break

# Display the image
row = next(iter(ds['train']))
row['tif'].show()
```

## ⚙️ Dataloader Options

### 🧩 `task` Options
Specify the type of information to load from the dataset:

- `binary`: Load only binary CAFO vs. non-CAFO labels
- `multi`: Load multiclass CAFO categories (e.g., swine, dairy, poultry)
- `bbox`: Load image, label, and bounding box of CAFO areas
- `infra`: Load image, label, and infrastructure annotations (e.g., barns, ponds)
- `all`: Load the full dataset with all available fields

---

### 📊 `dataset_name` Options
Choose which version of the dataset to load:

- `verified`: Only human-verified CAFO labels
- `set1`: Custom training/validation/test split 1
- `set2`: Custom training/validation/test split 2
- `merged`: Combined dataset from multiple sources
- `augmented`: Includes augmented image patches
---

![Example Image](example_image.PNG)



