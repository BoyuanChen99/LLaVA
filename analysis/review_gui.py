import argparse
import os
import json
from PIL import Image, ImageTk
import tkinter as tk


def parse_args():
    argparser = argparse.ArgumentParser(description="Run inference with the model.")
    argparser.add_argument("--model_name", type=str, default="llava-v1.5-7b", help="Name of the model to use for inference.")
    argparser.add_argument("--benchmark", type=str, default="pope", help="The benchmark of the experiment. eg: pope, chair, etc.")
    argparser.add_argument("--data_folder", type=str, default="../../data", help="Path to the data folder containing POPE dataset.")
    argparser.add_argument("--subset", type=str, default="coco", help="Subset of the POPE dataset to use.")
    argparser.add_argument("--subsplit", type=str, default="popular", help="Subsplit of the POPE dataset to use.")
    argparser.add_argument("--output_folder", type=str, default="../results", help="Folder to save the results.")
    return argparser.parse_args()


class ImageViewer(tk.Tk):
    def __init__(self, images_info):
        super().__init__()
        # Initiate the window
        self.title("POPE Results Viewer")
        self.geometry("2000x2300")  # Enormous and very tall window!
        self.images_info = images_info
        self.idx = 0

        # UI Components
        self.img_label = tk.Label(self)
        self.img_label.pack(pady=50)

        # Prediction correctness label
        self.correctness_label = tk.Label(self, font=("Helvetica", 48, "bold"))
        self.correctness_label.pack(pady=10)

        self.info_text = tk.Text(self, height=8, wrap=tk.WORD, font=("Helvetica", 32), padx=40, pady=40)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=80)

        # Next button on the bottom
        self.next_button = tk.Button(self, text="Next", command=self.show_next, font=("Helvetica", 32), height=3, width=16)
        self.next_button.pack(pady=50)

        self.show_image()

    def show_image(self):
        if self.idx >= len(self.images_info):
            self.img_label.config(image='', text='Done!')
            self.correctness_label.config(text='')
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "No more images.")
            self.next_button.config(state=tk.DISABLED)
            return

        info = self.images_info[self.idx]
        pil_img = info['image'].resize((1800, 1300))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img

        # Show correctness
        pred = str(info['model_output']).strip().lower()
        ans = str(info['answer']).strip().lower()
        if pred == ans:
            self.correctness_label.config(text="CORRECT", fg="green")
        else:
            self.correctness_label.config(text="WRONG", fg="red")

        # Display info
        self.info_text.delete("1.0", tk.END)
        meta = (
            f"Prompt: {info['prompt']}\n"
            f"VLM output: {info['model_output']}\n"
            f"Answer (Label): {info['answer']}\n"
            f"Image File: {info['image_file']}\n"
            f"Question ID: {info['question_id']}\n"
            f"Model ID: {info['model_id']}\n"
            + "-"*90
        )
        self.info_text.insert(tk.END, meta)

    def show_next(self):
        self.idx += 1
        self.show_image()



def main(args):
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
    with open(answers_file, "r") as f:
        answers = [json.loads(line) for line in f]

    images_info = []
    for result in results:
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
        sub_year = image_file.lower().split('_')[1] if 'coco' in image_file.lower() else ''
        image_path = os.path.join(args.data_folder, args.subset, sub_year, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        images_info.append({
            "question_id": question_id,
            "prompt": prompt,
            "model_output": model_output,
            "model_id": model_id,
            "image_file": image_file,
            "answer": answer,
            "image": image,
        })

    if images_info:
        app = ImageViewer(images_info)
        app.mainloop()
    else:
        print("No images to display.")



if __name__ == "__main__":
    args = parse_args()
    main(args)
