import argparse
import os
import json
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.ttk as ttk


def parse_args():
    argparser = argparse.ArgumentParser(description="Run inference with the model.")
    argparser.add_argument("--model_name", type=str, default="llava-v1.5-7b", help="Name of the model to use for inference.")
    argparser.add_argument("--benchmark", type=str, default="pope", help="The benchmark of the experiment. eg: pope, chair, etc.")
    argparser.add_argument("--data_folder", type=str, default="../../data", help="Path to the data folder containing POPE dataset.")
    argparser.add_argument("--subset", type=str, default="gqa", help="Subset of the POPE dataset to use. The three options are: coco, aokvqa, and gqa.")
    argparser.add_argument("--subsplit", type=str, default="popular", help="Subsplit of the POPE dataset to use.")
    argparser.add_argument("--subfolder", type=str, default="val2014", help="Only relevant for coco subset.")
    argparser.add_argument("--output_folder", type=str, default="../results", help="Folder to save the results.")
    argparser.add_argument("--temperature", type=float, default="1.0", help="The temperature used for inference.")
    return argparser.parse_args()


class ImageViewer(tk.Tk):
    def __init__(self, images_info):
        super().__init__()
        self.title("POPE Results Viewer")
        self.geometry("2000x2300")
        self.images_info = images_info
        self.idx = 0

        self.img_label = tk.Label(self)
        self.img_label.pack(pady=50)

        self.correctness_label = tk.Label(self, font=("Helvetica", 48, "bold"))
        self.correctness_label.pack(pady=10)

        self.info_text = tk.Text(self, height=8, wrap=tk.WORD, font=("Helvetica", 32), padx=40, pady=40)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=80)

        # Define tag styles for "YES" and "NO"
        self.info_text.tag_configure("yes", foreground="dark green", font=("Helvetica", 32, "bold"))
        self.info_text.tag_configure("no", foreground="dark red", font=("Helvetica", 32, "bold"))

        # Frame to hold navigation buttons
        nav_frame = tk.Frame(self)
        nav_frame.pack(pady=50)

        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.show_prev, font=("Helvetica", 32), height=3, width=16)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(nav_frame, text="Next", command=self.show_next, font=("Helvetica", 32), height=3, width=16)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.select_button = tk.Button(nav_frame, text="Select Image", command=self.open_selector, font=("Helvetica", 32), height=3, width=16)
        self.select_button.pack(side=tk.LEFT, padx=10)

        self.show_image()


    def show_image(self):
        if self.idx >= len(self.images_info):
            self.img_label.config(image='', text='Done!')
            self.correctness_label.config(text='')
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert(tk.END, "No more images.")
            self.next_button.config(state=tk.DISABLED)
            return

        self.next_button.config(state=tk.NORMAL)
        self.prev_button.config(state=tk.NORMAL if self.idx > 0 else tk.DISABLED)

        info = self.images_info[self.idx]
        pil_img = info['image'].resize((1800, 1300))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img)
        self.img_label.image = self.tk_img

        pred = str(info['model_output']).strip().lower()
        ans = str(info['answer']).strip().lower()
        if pred == ans:
            self.correctness_label.config(text="CORRECT", fg="green")
        else:
            self.correctness_label.config(text="WRONG", fg="red")

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

        # Highlight yes/no
        for tag_text, tag_name in [("yes", "yes"), ("no", "no")]:
            start = "1.0"
            while True:
                pos = self.info_text.search(tag_text, start, tk.END, nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(tag_text)}c"
                self.info_text.delete(pos, end)
                self.info_text.insert(pos, tag_text.upper(), tag_name)
                start = f"{pos}+{len(tag_text)}c"

    def show_next(self):
        if self.idx < len(self.images_info) - 1:
            self.idx += 1
            self.show_image()

    def show_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_image()

    def open_selector(self):
        selector = tk.Toplevel(self)
        selector.title("Select Image")
        selector.geometry("600x800")

        # Create a custom style for a wider scrollbar handle
        style = ttk.Style(selector)
        style.theme_use("default")
        style.configure("Vertical.TScrollbar", gripcount=0, width=30, height=200, troughcolor='lightgray', bordercolor='gray', arrowcolor='black')

        # Use ttk Scrollbar (with larger handle)
        # Create custom scrollbar style with larger dimensions
        style = ttk.Style(selector)
        style.theme_use("default")

        # Wider scrollbar and bigger handle area
        style.configure("Vertical.TScrollbar",
            gripcount=1,
            width=40,                # <-- Wider scrollbar
            troughcolor='lightgray',
            background='darkgray',
            bordercolor='gray',
            arrowcolor='black'
        )

        # Taller scrollbar "thumb" (the draggable handle)
        style.layout("Vertical.TScrollbar",
            [
                ("Vertical.Scrollbar.trough", {
                    "children": [
                        ("Vertical.Scrollbar.thumb", {
                            "expand": "1",
                            "sticky": "nswe"
                        })
                    ],
                    "sticky": "ns"
                })
            ]
        )

        # Use this styled scrollbar
        scrollbar = ttk.Scrollbar(selector, orient=tk.VERTICAL, style="Vertical.TScrollbar")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Larger font for listbox entries
        listbox_font = ("Helvetica", 32)
        listbox = tk.Listbox(selector, font=listbox_font, yscrollcommand=scrollbar.set, width=200)
        for i, info in enumerate(self.images_info):
            clean_file = os.path.splitext(info['image_file'])[0]  # Strip ".jpg"
            item = f"QID: {info['question_id']} | File: {clean_file}"
            listbox.insert(tk.END, item)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        def on_select(event):
            selection = listbox.curselection()
            if selection:
                self.idx = selection[0]
                self.show_image()
                selector.destroy()

        listbox.bind("<<ListboxSelect>>", on_select)




def main(args):
    ### Step 0.1: Define, and read the output file
    if args.temperature <= 0:
        temperature = "greedy"
    else:
        temperature = str(args.temperature)
        if temperature.endswith(".0"):
            temperature = "temp"+temperature[:-2]
    output_file = os.path.join(args.output_folder, args.benchmark, temperature, f"{args.subset}_pope_{args.subsplit}_{args.model_name}.jsonl")
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist. Please run the inference script first.")
        return
    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f.readlines()]

    ### Step 0.2: Define, and read the answers file
    answers_file = os.path.join(args.data_folder, args.benchmark.upper(), args.subset, f"{args.subset}_{args.benchmark.lower()}_{args.subsplit}.json")
    if not os.path.exists(answers_file):
        print(f"Answers file {answers_file} does not exist. Please check the path.")
        return
    with open(answers_file, "r") as f:
        answers = [json.loads(line) for line in f]

    ### Step 0.3: Define image folder
    args.subset = args.subset.lower()
    if "gqa" in args.subset:
        image_dir = os.path.join(args.data_folder, args.subset, "allImages", "images")
    elif "coco" in args.subset or "aokvqa" in args.subset:
        image_dir = os.path.join(args.data_folder, args.subset, args.subfolder)


    ### Step 1: Loop through the images in the GUI
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
        image_path = os.path.join(image_dir, image_file)
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
