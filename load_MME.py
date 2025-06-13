"""
MME dataset loader from huggingface repo released by the official team. 
The dataset has only one split: 'test', and the length is 2374. 
The dataset has 5 keys: 'question_id', 'image', 'question', 'answer', 'category'. 
Each image is a PIL image object. Its dimension and format (.png, .jpg, etc.) is not fixed. 
All 'answer' values are either 'Yes' or 'No'.
"""

from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import random

# Load the dataset
ds = load_dataset("lmms-lab/MME")
print("Available splits:", ds.keys())
split = "test" if "test" in ds else list(ds.keys())[0]
dataset_length = len(ds[split])
print(f"Dataset split '{split}' contains {dataset_length} examples.")
print("\nKeys:", ds[split][0].keys())
sequential_indices = list(range(dataset_length))
random_indices = random.sample(sequential_indices, 5)

# Get all unique values from 'answer' field
unique_answers = set(example["answer"] for example in ds[split])
print(f"\nUnique answers in the dataset: {unique_answers}")

# Following the way above, log the dataset, line by line, with image plotted
for i in random_indices:
    example = ds[split][i]
    print(f"\nExample {i} metadata:", {k: example[k] for k in example if k != "image"})
    if "image" in example:
        img = example["image"]
        # Get the dimension of this img
        img_size = img.size if isinstance(img, Image.Image) else None
        print(type(img))
        if isinstance(img, Image.Image):
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Example {i} Image")
            plt.show()
        else:
            print(f"Image field for example {i} is not a PIL image:", type(img))
    else:
        print(f"No image found for example {i}.")
        