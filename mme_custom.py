from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
import argparse
import os
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="baseline")
parser.add_argument("--tau", type=float, default="-0.3")
parser.add_argument("--beta_1", type=float, default="0.05")
parser.add_argument("--beta_2", type=float, default="0.20")
parser.add_argument("--alpha", type=float, default="0.7")
args1 = parser.parse_args()
output_dir = args1.output_dir


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory {output_dir} already exists.")


    for item in folders_and_files:
        image_folder = item["image_folder"]
        output_file_path = item["output_file_path"]

        with open(output_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            idx = 0
            for line in lines:
                print(f"Processing {idx} in {output_file_path}")
                idx += 1
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_file = parts[0]
                    question = parts[1]
                    image_path = os.path.join(image_folder, image_file)
                    args = type('Args', (), {
                        "query": question,
                        "conv_mode": None,
                        "image_file": image_path,
                        "sep": ",",
                        "temperature": 0,
                        "top_p": None,
                        "num_beams": 1,
                        "max_new_tokens": 512,
                        "device_map":"cuda",
                        "device":"cuda",
                        "model": model,
                        "tokenizer": tokenizer,
                        "image_processor":image_processor,
                        "context_len":context_len,
                        "model_name": get_model_name_from_path(model_path),
                        "output_hidden_states": True,   ### This is necessary for DAMO.
                        "tau":args1.tau,
                        "beta_1":args1.beta_1,
                        "beta_2":args1.beta_2,
                        "alpha":args1.alpha,
                        
                    })()
                    output_text = eval_model(args)
                    output_text = output_text.replace('\n', ' ')
                    output_text = output_text.replace('\t', ' ')
                    output_line = f"{line.strip()}\t{output_text}\n"
                    output_file.write(output_line)

        print(f"Finished processing {output_file_path}")

    print("All files have been processed!")

    subprocess.run(["python", "calculation.py", "--results_dir", output_dir], cwd=eval_tool_dir)

