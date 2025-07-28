import os
import json
import numpy as np
import argparse

argparse = argparse.ArgumentParser(description="Generate a JSON file for the CHAIR dataset.")
argparse.add_argument("--coco_dir", type=str, default="../data/coco/val2014", help="Path to the MSCOCO directory")
argparse.add_argument("--chair_dir", type=str, default="../data/CHAIR", help="Path to the output CHAIR directory")
argparse.add_argument("--seed", type=int, default=1994, help="Random seed for numpy")
FLAGS = argparse.parse_args()


def main(args):
    ### Step 1: Randomly select 500 filenames ending with .jpg from the MSCOCO dir, with numpy
    np.random.seed(args.seed)
    all_files = [f for f in os.listdir(args.coco_dir) if f.endswith('.jpg')]
    selected_files = np.random.choice(all_files, size=500, replace=False)
    selected_files = sorted(selected_files)

    ### Step 2: One-by-one create dict entries and write to a .json file
    for question_id in range(1, len(selected_files)+1):
        image_file = selected_files[question_id-1]
        question = {
            "question_id": question_id,
            "image": image_file,
            "text": f"Please help me describe the image in detail.",
            "seed": args.seed
        }
        with open(os.path.join(args.chair_dir, f"chair_{args.seed}.json"), "a") as f:
            json.dump(question, f)
            f.write("\n")



if __name__ == "__main__":
    main(FLAGS)