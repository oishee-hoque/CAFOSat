import yaml
from classify.classifier import CafoBinaryClassifier

def main():
    with open("classify_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    cfg = config["inference"]

    # Format dynamic path
    input_dir = cfg["input_dir"].format(type=cfg["type"])
    output_dir = cfg["output_dir"].format(type=cfg["type"])

    classifier = CafoBinaryClassifier(cfg["checkpoint"])
    classifier.classify_folder(
        input_dir=input_dir,
        output_dirs={
            "cafo": f"{output_dir}/cafo",
            "non_cafo": f"{output_dir}/non_cafo"
        }
    )
    
    stats = classifier.classify_folder(
    input_dir=input_dir,
    output_dirs={
        "cafo": f"{output_dir}/cafo",
        "non_cafo": f"{output_dir}/non_cafo"
    }
)

    # Optionally log or write to file
    print(f"\n✅ Completed: {stats['cafo']} CAFO, {stats['non_cafo']} Non-CAFO")


if __name__ == "__main__":
    main()
