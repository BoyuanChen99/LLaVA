"""
This pope evaluation script is adapted from predict.py in original LLaVA-v1.5 repo, and the AGLA code repo. As the latter does not support multi-gpu, it is very likely to exceed memory limit. Therefore, this script is created. 
"""

import os
import torch
from PIL import Image
import requests
from io import BytesIO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from cog import BasePredictor, Input, Path
import json
from tqdm import tqdm

### The line commented out is from the original predict.py. It is not necessary, as users mostly have their own HF cache directory.
# os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"
HUGGINGFACE_CACHE = os.getenv("HF_HOME")

### Get the hf token from env, as LLaVA-v1.5 requires it to download the model weights
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

### Load the model and tokenizer
# url for the weights mirror (Already dead, so need to be replaced)
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
HF_WEIGHTS_URL = "https://huggingface.co"
model_name = "llava-v1.5-7b"
model_author = "liuhaotian"
# files to download from the weights mirrors
weights = [
    {
        "dest": HUGGINGFACE_CACHE + f"/{model_author}/{model_name}",
        # git commit hash from huggingface (obsolete, as the LLaVA-v1.5 is no longer updated, we don't need to use git commit hash)
        # "src": "llava-v1.5-7b/12e054b30e8e061f423c7264bc97d4248232e965",
        "src": f"{model_author}/{model_name}/resolve/main",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            # "pytorch_model-00003-of-00003.bin", # 3 is only used for 13B model
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        # The visual encoder is the same for both 7B and 13B models
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "openai/clip-vit-large-patch14-336/resolve/main",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]

def download_json(url: str, dest: Path, hf_token: str):
    print(f"Inside download_json: {url} to {dest}")
    headers = {"Authorization": f"Bearer {hf_token}"}
    res = requests.get(url, allow_redirects=True, headers=headers)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

### Our own custom download function with token support
def download_file_with_token(url, dest: Path, token: str):
    print(f"⬇ Downloading {url} → {dest}")
    headers = {"Authorization": f"Bearer {token}"}
    if dest.exists():
        print(f"File {dest} already exists. Skipping download.")
    else:
        with requests.get(url, headers=headers, stream=True) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Failed to download {url}. Status code: {r.status_code}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        for weight in weights:
            # download_weights(weight["src"], weight["dest"], weight["files"])
            for file in weight["files"]:
                download_file_with_token(
                    url = os.path.join(HF_WEIGHTS_URL, weight["src"], file),
                    dest = Path(weight["dest"], file),
                    token = HF_TOKEN
                )
        disable_torch_init()
    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(f"{model_author}/{model_name}", model_name=model_name, model_base=None, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens", ge=0.0, le=1.0, default=1.0),
        top_k: int = Input(description="When decoding text, samples from the top k most likely tokens", ge=1, default=None),
        temperature: float = Input(description="Adjusts randomness of outputs", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate", default=1024, ge=0),
    ) -> str:
        """Run a single prediction on the model (non-streaming)"""

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        # Process image
        # image = "./examples/cat.png"
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

        # Construct input prompt
        # prompt = "What is in this image? What is the estimated weight of the subject?"
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Input IDs: {input_ids}")
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.inference_mode():
            if temperature > 0:
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_tokens,
                    use_cache=True
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                    use_cache=True
                )
        input_token_len = input_ids.shape[1]
        output_text = self.tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0].strip()

        # Remove stop string if present
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)].strip()

        return output_text

    
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


### ======== Main ======== ###
def main():
    ### Step 0: Initialize the VLM
    predictor = Predictor()
    predictor.setup()
    
    ### Step 1: Load CHAIR
    data_folder = "../data"
    seed = 1994
    model_name = "llava-v1.5-7b"
    output_folder = "./results/chair"
    os.makedirs(output_folder, exist_ok=True)

    questions = [json.loads(q) for q in open(f"{data_folder}/CHAIR/chair_{seed}.json", "r")]
    output_file = f"{output_folder}/chair_{seed}_{model_name}.jsonl"

    ### Step 2: Loop through the questions
    for line in tqdm(questions):
        ### Step 2.1: Initialize the necessary variables
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        cur_prompt = question

        ### !!!I'm not sure why we need to append this, because this will make one prompt have two image tokens (<image>), and the llava's original code throws error. I need to check other works' code on llava exps. 
        # if predictor.model.config.mm_use_im_start_end: 
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n'  + question
        qs = question

        ### Step 2.2: Load the image
        image_path = Path(os.path.join(data_folder, "coco", "val2014", image_file))


        ### Step 3: Generate the answer with predictor. Parameters are set to match the CHAIR protocol.
        output = predictor.predict(
            image=image_path,
            prompt=qs,
            temperature=0.0, 
            max_tokens=512,
        )
        # print(output)


        ### Step 4: Save the output to the output file
        with open(output_file, "a") as f:
            f.write(json.dumps({"question_id": idx, 
                                "prompt": cur_prompt,
                                "text": output.strip(), 
                                "model_id": model_name,
                                "image": image_file, 
                                "metadata": {}}) + "\n")



if __name__ == "__main__":
    main()
