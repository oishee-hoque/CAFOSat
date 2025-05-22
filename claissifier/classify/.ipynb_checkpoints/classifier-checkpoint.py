# classify/cafo_binary_classifier.py

import os
import shutil
import torch
from collections import Counter
from classify.utils import load_image, ensure_dir, is_image_file
from model.classifierModel import CAFOClassifier

class CafoBinaryClassifier:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CAFOClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

    def classify_folder(self, input_dir, output_dirs):
        stats = Counter()

        ensure_dir(output_dirs["cafo"])
        ensure_dir(output_dirs["non_cafo"])

        for filename in os.listdir(input_dir):
            if not is_image_file(filename):
                continue

            path = os.path.join(input_dir, filename)
            try:
                image_tensor = load_image(path).to(self.device)

                with torch.no_grad():
                    output = self.model(image_tensor)
                    pred = torch.argmax(output, dim=1).item()

                label = "cafo" if pred == 1 else "non_cafo"
                shutil.copy(path, os.path.join(output_dirs[label], filename))
                stats[label] += 1

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                stats["error"] += 1

        print("\n🔢 Classification Summary:")
        print(f"  CAFO:     {stats['cafo']}")
        print(f"  Non-CAFO: {stats['non_cafo']}")
        if stats["error"]:
            print(f"  Errors:   {stats['error']}")

        return stats

    def classify_categories(self, base_input_dir, output_base_dir, categories):
        for category in categories:
            input_dir = os.path.join(base_input_dir, category)
            output_dirs = {
                "cafo": os.path.join(output_base_dir, category, "CAFO"),
                "non_cafo": os.path.join(output_base_dir, category, "NON-CAFO")
            }

            print(f"🔍 Processing category: {category}")
            self.classify_folder(input_dir, output_dirs)

    def reclassify_non_cafo(self, base_verified_dir, category):
        from PIL import Image
        from torchvision import transforms

        non_cafo_dir = os.path.join(base_verified_dir, category, "NON-CAFO")
        cafo_dir = os.path.join(base_verified_dir, category, "CAFO")
        ensure_dir(cafo_dir)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        new_cafo = []
        for filename in os.listdir(non_cafo_dir):
            path = os.path.join(non_cafo_dir, filename)
            if not is_image_file(filename):
                continue
            try:
                image = Image.open(path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = torch.argmax(self.model(tensor)).item()

                if pred == 1:
                    new_path = os.path.join(cafo_dir, filename)
                    shutil.move(path, new_path)
                    new_cafo.append(new_path)
            except Exception as e:
                print(f"Error reclassifying {filename}: {e}")

        return new_cafo
