## 🧠 Classifier Training

This module supports both **binary classification** (e.g., CAFO vs. non-CAFO) and **multi-class classification** (e.g., different CAFO types).

---

### 🛠️ Dependencies

- `torchvision` – for model backbones like ResNet  
- `pytorch-lightning` – for managing training and logging  
- `scikit-learn` – for evaluation metrics  
- `tqdm`, `pandas`, `numpy`, etc. – for data handling and progress tracking

✅ All of these are already included in the `requirements.txt` file in data_processing.

#### ⚙️ Configuration

All training settings are controlled via a YAML config file located in the `config/` directory.  
Example: `config/classifier_config_iowa.yaml`

### 🗂 Dataset Selection

We provide multiple training splits for experimentation and evaluation:

- `../classifier/datas/dataset/cafosat_set1_training.json`
- `../classifier/datas/dataset/cafosat_set2_training.json`
- `../classifier/datas/dataset/cafosat_verified_training.json`
- `../classifier/datas/dataset/cafosat_augmented_training.json`
- `../classifier/datas/dataset/cafosat_merged_training.json`
- `../classifier/datas/dataset/cafosat_all_training.json`
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

🔧 Update Config: Before running evaluation, update `config` in the files to point to the correct evaluation dataset.


## 📊 Model Performance Comparison

The following results were obtained by training and evaluating various backbone models on the **CAFOSat** dataset using two validation sets:

- **Val Set 1**: `cafosat_set1_training.json`
- **Val Set 2**: `cafosat_set2_training.json`

These sets represent distinct splits designed to test the models’ generalization performance across different subsets of CAFO imagery.

---

### 🔍 Evaluation Metrics

- **Accuracy (Acc.)**: Overall classification accuracy
- **F1 Score (F1)**: Harmonic mean of precision and recall
- **Mean Average Precision (mAP)**: Averaged precision across all classes

---

### 📈 Results Table

| Model        | Acc. (Set 1) | F1 (Set 1) | mAP (Set 1) | Acc. (Set 2) | F1 (Set 2) | mAP (Set 2) |
|--------------|--------------|------------|-------------|--------------|------------|-------------|
| ResNet18     | 0.466        | 0.413      | 0.401       | 0.503        | 0.452      | 0.415       |
| ResNet50     | 0.502        | 0.453      | **0.444**   | 0.508        | 0.454      | 0.414       |
| ViT          | 0.478        | 0.424      | 0.374       | 0.491        | 0.436      | 0.394       |
| Swin         | **0.531**    | **0.485**  | 0.443       | 0.482        | 0.437      | 0.419       |
| ConvNeXt     | 0.528        | 0.483      | 0.412       | 0.491        | 0.440      | 0.403       |
| EfficientNet | 0.526        | 0.470      | 0.431       | **0.539**    | **0.479**  | **0.420**   |
| CLIP         | 0.375        | 0.314      | 0.298       | 0.370        | 0.328      | 0.313       |
| RemoteCLIP   | 0.434        | 0.379      | 0.342       | 0.400        | 0.354      | 0.310       |
| DinoV2       | 0.511        | 0.453      | 0.371       | 0.468        | 0.433      | 0.404       |

---

