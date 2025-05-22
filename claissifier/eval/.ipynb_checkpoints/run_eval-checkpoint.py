import os
import pandas as pd
from eval_cafo_only import predict_folder
import sys

log_file_path = "eval_log.txt"  # Change this path as needed
sys.stdout = open(log_file_path, 'w')  # Redirect stdout to file

folders = os.listdir('/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/point_single_patches')
state_list = set()
for f in folders:
    base = f.split('_')[0]  # e.g., 'DE' from 'DE_filtered'
    if base: state_list.add(base)
    
    
# state_list = ['DE']

base_input_dir = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/point_single_patches"
base_output_dir = "/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/data_preparation/datas/misclassified_cafos"

for state in state_list:
    print(f"\n🚀 Running evaluation for {state}")
    config = {
        'checkpoint_path': '/project/biocomplexity/gza5dr/CAFO/exp_v2/main_experiments/cafo_classification/checkpoints/cafo-best-epoch=02-val_acc=0.99_resnet50_IOWA.ckpt',
        'image_folder': os.path.join(base_input_dir, f"{state}_filtered"),
        'output_csv': os.path.join(base_output_dir, f"filtered_misclassified_patches_{state}.csv"),
        'output_csv_classified': os.path.join(base_output_dir, f"filtered_classified_patches_{state}.csv"),
        'input_size': (224, 224),
        'device': 'cuda',
        'cafo_class_id': 1
    }

    predict_folder(config)
