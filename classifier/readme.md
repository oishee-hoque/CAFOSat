## 🧠 Classifier Training

This module supports both **binary classification** (e.g., CAFO vs. non-CAFO) and **multi-class classification** (e.g., different CAFO types).

---

#### ⚙️ Configuration

All training settings are controlled via a YAML config file located in the `config/` directory.  
Example: `config/classifier_config_iowa.yaml`

### 🗂 Dataset Selection

We provide multiple training splits for experimentation and evaluation:

- `../classifier/datas/dataset/cafosat_set1_training.json`
- `../classifier/datas/dataset/cafosat_set2_training.json`
- `../classifier/datas/dataset/cafosat_set3_training.json`
- `../classifier/datas/dataset/cafosat_set4_training.json`
- `../classifier/datas/dataset/cafosat_set5_training.json`
- `../classifier/datas/dataset/cafosat_set6_training.json`
- `../classifier/datas/dataset/cafosat_set7_training.json`
- `../classifier/datas/dataset/nc_training_data.json`
- `../classifier/datas/dataset/train_test_split_meter_ml.json`

These represent different randomized or stratified splits of the CAFOSat dataset.

##### 📥 Download Required
We also support training on additional datasets beyond CAFOSat, such as:

- `nc_training_data.json` – from the NC-CAFO dataset
- `train_test_split_meter_ml.json` – from the Meter ML dataset

To use these datasets, please first **download the raw data** from their respective repositories:

- 🔗 **Meter ML**: [https://github.com/ProjectDrawdown/MeterML](https://github.com/ProjectDrawdown/MeterML)
- 🔗 **RegLab CAFO Dataset**: [https://github.com/reglab/cafo-detection](https://github.com/reglab/cafo-detection)

---

### 🔧 How to Use a Dataset Split

To use any of the above splits, update the `data_path` field in your config file:

```yaml
# === Data ===
data_path: /project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/datas/dataset/cafosat_set1_training.json
```

#### ▶️ Running Binary Classifier

```bash
python classifier_train.py --config_file config/classifier_config_set1.yaml
```

#### ▶️ Running Binary Classifier
```bash
python multi_train.py --config_file config/classifier_config_set1.yaml

```


## 📊 Model Evaluation

You can evaluate trained classifiers on any dataset using the provided evaluation scripts.  
The input dataset should be in **JSON format** and follow the same structure used for training.

---

### 🧪 Evaluation Scripts

| Script                | Description                                     |
|-----------------------|-------------------------------------------------|
| `eval_cafo_only.py`   | For **binary classification** (CAFO vs. non-CAFO) |
| `eval_v3.py`          | For **multi-class classification** (e.g., CAFO types) |

---

### 🔧 Update Config

Before running evaluation, update your `config` to point to the correct evaluation dataset:

```yaml
# === Data ===
data_path: path/to/your_eval_dataset.json  # e.g., cafosat_set7_training.json or meter_ml.json
