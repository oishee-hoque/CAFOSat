import subprocess
import yaml

def load_config(yaml_path="config_v3.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def run_step(script_name, label, yaml_path):
    print(f"Running {label}...")
    result = subprocess.run(["python", script_name, yaml_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{label} completed successfully.")
    else:
        print(f"{label} failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

def main(yaml_path="config.yaml"):
    config = load_config(yaml_path)
    run_flags = config['run_flags']
    print(run_flags)

    if run_flags.get('match_point_to_image'):
        run_step("matchPointToImage.py", "matchPointToImage", yaml_path)

    if run_flags.get('multi_patch_generate'):
        run_step("multiplePatchGeneration.py", "multiplePatchGeneration", yaml_path)

    if run_flags.get('single_patch_generate'):
        run_step("singlePatchGeneration.py", "singlePatchGeneration", yaml_path)

    if run_flags.get('refine_coords'):
        run_step("refiningCoords.py", "refiningCoords", yaml_path)

    if run_flags.get('cluster'):
        run_step("clustering.py", "clustering", yaml_path)

    if run_flags.get('single_patch_filtered'):
        run_step("singlePatchGenerationfiltered.py", "singlePatchGenerationFiltered", yaml_path)



if __name__ == "__main__":
    main(yaml_path="config.yaml")
