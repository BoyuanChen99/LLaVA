import argparse
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(description="Run inference with the model.")
### VLM info
argparser.add_argument("--model_name", type=str, default="llava-v1.5-7b", help="Name of the model to use for inference.")
### Image set
argparser.add_argument("--benchmark", type=str, default="pope", help="The benchmark of the experiment. eg: pope, chair, etc.")
argparser.add_argument("--data_folder", type=str, default="../../data", help="Path to the data folder containing POPE dataset.")
argparser.add_argument("--subset", type=str, default="coco", help="Subset of the POPE dataset to use.")
argparser.add_argument("--subsplit", type=str, default="popular", help="Subsplit of the POPE dataset to use.")
argparser.add_argument("--output_folder", type=str, default="../results", help="Folder to save the results.")
FLAGS = argparser.parse_args()


def main(args):
    ### Step 0: Read the output .jsonl file, and the answer file
    output_file = os.path.join(args.output_folder, f"{args.subset}_pope_{args.subsplit}_{args.model_name}.jsonl")
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist. Please run the inference script first.")
        return
    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]
    answers_file = os.path.join(args.data_folder, args.benchmark.upper(), args.subset, f"{args.subset}_{args.benchmark.lower()}_{args.subsplit}.json")
    if not os.path.exists(answers_file):
        print(f"Answers file {answers_file} does not exist. Please check the path.")
        return
    print(f"Answers file is: {answers_file}")
    with open(answers_file, "r") as f:
        answers = [json.loads(line) for line in f]


    ### Step 1: Loop through the results
    for result in results:
        ### Step 1.1: Extract the necessary information
        question_id = result["question_id"]
        prompt = result["prompt"]
        model_output = result["text"].lower().strip()
        model_id = result.get("model_id", args.model_name)
        image_file = result["image"]
        answer = next(
            (a["label"] for a in answers 
            if a["question_id"] == question_id and a["text"] == prompt and a["image"] == image_file),
            "NOT FOUND"
        )

        ### Step 1.2: Load and plot the image
        sub_year = image_file.lower().split('_')[1] if 'coco' in image_file.lower() else ''
        image_path = os.path.join(args.data_folder, args.subset, sub_year, image_file)
        image = Image.open(image_path).convert("RGB")

        ### Step 1.3: Log info and show image with plt
        print(f"Question ID: {question_id}")
        print(f"Prompt: {prompt}")
        print(f"VLM output: {model_output}")
        print(f"Image File: {image_file}")
        print(f"Model ID: {model_id}")
        print("-" * 40)
        plt.imshow(image)
        plt.title(f"Question ID: {question_id}\nPrompt: {prompt}\nVLM Output: {model_output}\nAnswer(Label): {answer}")
        plt.axis("off")
        plt.show()



if __name__ == "__main__":
    main(FLAGS)
