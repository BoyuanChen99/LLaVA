"""
Adapted from ./predict.py in original LLaVA =repo.
"""
import os
import torch
from PIL import Image
import requests
from io import BytesIO
import time
import subprocess
from threading import Thread
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer
from cog import BasePredictor, Input, Path, ConcatenateIterator

# This line is from the original predict.py, but it is not necessary, as users mostly have their own HF cache directory.
# os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

### Get the hf token from env, as LLaVA-v1.5 requires it to download the model weights
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

### Load the model and tokenizer
# url for the weights mirror (Already dead, so need to be replaced)
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
HF_WEIGHTS_URL = "https://huggingface.co"
# files to download from the weights mirrors
weights = [
    {
        "dest": "liuhaotian/llava-v1.5-7b",
        # git commit hash from huggingface (obsolete, as the LLaVA-v1.5 is no longer updated, we don't need to use git commit hash)
        # "src": "llava-v1.5-7b/12e054b30e8e061f423c7264bc97d4248232e965",
        "src": "liuhaotian/llava-v1.5-7b/resolve/main",
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

# Originally from predict.py, but it is no longer working, as pget does not support verified downloads.
def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(HF_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest, HF_TOKEN)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)

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
    
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model("liuhaotian/llava-v1.5-13b", model_name="llava-v1.5-13b", model_base=None, load_8bit=False, load_4bit=False)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
        top_p: float = Input(description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens", ge=0.0, le=1.0, default=1.0),
        temperature: float = Input(description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic", default=0.2, ge=0.0),
        max_tokens: int = Input(description="Maximum number of tokens to generate. A word is generally 2-3 tokens", default=1024, ge=0),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
    
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()
    
        # loop start
    
        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
    
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=dict(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True))
            thread.start()
            # workaround: second-to-last token is always " "
            # but we want to keep it if it's not the second-to-last token
            prepend_space = False
            for new_text in streamer:
                if new_text == " ":
                    prepend_space = True
                    continue
                if new_text.endswith(stop_str):
                    new_text = new_text[:-len(stop_str)].strip()
                    prepend_space = False
                elif prepend_space:
                    new_text = " " + new_text
                    prepend_space = False
                if len(new_text):
                    yield new_text
            if prepend_space:
                yield " "
            thread.join()
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main():
    predictor = Predictor()
    predictor.setup()
    
    # Example usage
    image_path = "./examples/cat.png"  # Replace with your image path
    prompt = "What is in this image? What is the estimated weight of the subject?"
    
    for output in predictor.predict(
        image=Path(image_path),
        prompt=prompt,
        top_p=0.95,
        temperature=0.7,
        max_tokens=200,
    ):
        print(output, end='', flush=True)



if __name__ == "__main__":
    main()