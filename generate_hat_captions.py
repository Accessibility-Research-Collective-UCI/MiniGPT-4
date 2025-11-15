from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.common.config import Config
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from PIL import Image
import io
from io import BytesIO
import base64
import requests

import time
from datetime import datetime
import copy
import os
import traceback

import torch
import json


class MiniGPT4Captioner:
    def __init__(self, prompt="Describe this image."):
        self.prompt = prompt

    # ------------------- Image Processing -------------------
    def remove_transparency(self, im, bg_colour=(255, 255, 255)):
        """
        Remove transparency from an image.

        Args:
            im (PIL.Image.Image): Image to remove transparency from.
            bg_colour (tuple, optional): Background color to use for the transparent areas. Defaults to (255, 255, 255).

        Returns:
            PIL.Image.Image: Image with transparency removed.
        """
        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = im.convert("RGBA").split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", im.size, bg_colour + (255,))
            bg.paste(im, mask=alpha)
            return bg

        else:
            return im

    def convert_to_base64(self, image_url):
        """
        Convert an image, specified by its url, to a PNG and return the base64 encoded string.

        Args:
            image_url (str): URL of the image to convert.

        Returns:
            str: Base64 encoded string of the image.
        """
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # remove transparency
        image = self.remove_transparency(image)

        with BytesIO() as f:
            image.save(f, format="PNG")
            f.seek(0)

            return base64.b64encode(f.read()).decode("utf-8")

    def convert_to_png(self, image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # remove transparency
        image = self.remove_transparency(image)

        with BytesIO() as f:
            image.save(f, format="PNG")
            return f.getvalue()

    # ------------------- Caption Generation -------------------
    def get_minigpt_caption(
        self, image_url, model, processor, temperature=1.0, top_p=0.95
    ):
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.system = ""
        model.eval()

        img = Image.open(io.BytesIO(self.convert_to_png(image_url)))

        if img.mode == "L":
            print("Wrong mode!")
            img = img.convert("RGB")
        img = torch.unsqueeze(processor(img), 0)

        texts = prepare_texts([self.prompt], conv_temp)

        answers = model.generate(
            img,
            texts,
            max_new_tokens=500,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        return answers[0]

    def load_minigpt_model(self, args):
        return init_model(args)

    def generate_dataset(
        self,
        args,
        dataset,
        save_folder="/home/ngtj/MiniGPT-4/data/",
        file_prefix="minigpt",
        model_key="minigpt",
        temp=1.0,
        top_p=0.95,
        half=None,
    ):
        print("Starting to caption...", flush=True)
        out_json = copy.deepcopy(dataset)
        tmp_dir = os.path.join(save_folder, "tmp")
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)

        model, vis_processor = self.load_minigpt_model(args)

        print("Done loading model.", flush=True)
        try:
            for i, dat in enumerate(dataset):
                if half == 1 and i > len(dataset) // 2:
                    continue
                elif half == 2 and i <= len(dataset) // 2:
                    continue

                # minigpt messed up
                if i < 689:
                    continue

                print(f"{i}: {dat['file_name']}", flush=True)

                # generate reference
                print("\tGenerating ref...")
                start = time.perf_counter()
                cap = self.get_minigpt_caption(
                    model=model,
                    processor=vis_processor,
                    image_url=dat["image_url"],
                    temperature=temp,
                    top_p=top_p,
                )
                print(f"\t{cap}")
                out_json[i]["captions"][model_key]["reference_caption"] = cap
                print(f"\tDone in {time.perf_counter() - start} s.", flush=True)

                # generate samples
                print("\tGenerating samples...")
                for sample_idx in range(10):
                    print(f"\t\tSample {sample_idx + 1}...")
                    start = time.perf_counter()
                    cap = self.get_minigpt_caption(
                        model=model,
                        processor=vis_processor,
                        image_url=dat["image_url"],
                        temperature=temp,
                        top_p=top_p,
                    )
                    print(f"\t\t{cap}")

                    out_json[i]["captions"][model_key]["samples"][sample_idx] = cap
                    print(f"\t\tDone in {time.perf_counter() - start} s.", flush=True)

                # write every 50 iters?
                if i % 50 == 0 and i > 0:
                    date = datetime.now()
                    tmp_name = f"cached_{file_prefix}_{half}_{date.year:04}{date.month:02}{date.day:02}_{date.hour:02}{date.minute:02}{date.second:02}.json"
                    with open(os.path.join(tmp_dir, tmp_name), "w") as f:
                        f.write(json.dumps(out_json, indent=2))
                    print(
                        f"Intermediate file {tmp_name} saved to {tmp_dir}", flush=True
                    )

        except Exception:
            print(f"Error: {traceback.format_exc()}")
        finally:
            # write json to file
            date = datetime.now()
            outfile_name = f"{file_prefix}_{half}_{date.year:04}{date.month:02}{date.day:02}_{date.hour:02}{date.minute:02}{date.second:02}.json"
            with open(os.path.join(save_folder, outfile_name), "w") as f:
                f.write(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    """
    python /home/ngtj/MiniGPT-4/test_minigpt.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
    """
    parser = eval_parser()
    args = parser.parse_args()
    captioner = MiniGPT4Captioner()
    cfg = Config(args)

    # read HAT
    print("Reading HAT dataset...", flush=True)
    hat_file = "./data/hat_empty_prepared.json"

    with open(hat_file, "r", encoding="utf-8") as f:
        hat_data = json.load(f)

    print("Success!", flush=True)

    # make all the captions
    captioner.generate_dataset(args, hat_data, half=1)
