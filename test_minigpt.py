from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.common.config import Config
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from PIL import Image
import io
from io import BytesIO
import requests

import torch

def remove_transparency(im, bg_colour=(255, 255, 255)):
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

def convert_to_png(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # remove transparency
    image = remove_transparency(image)

    with BytesIO() as f:
        image.save(f, format="PNG")
        return f.getvalue()

if __name__ == "__main__":
    parser = eval_parser()
    args = parser.parse_args()
    # args = ["--cfg-path eval_configs/minigpt4_llama2_eval.yaml",
    #         "--gpu-id 0"]
    '''
    python /home/ngtj/MiniGPT-4/test_minigpt.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
    '''
    cfg = Config(args)
    
    model, vis_processor = init_model(args)
    # conversation template
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    model.eval()

    prompt = "Describe this image."
    image_url = "http://images.cocodataset.org/val2014/COCO_val2014_000000176017.jpg"
    img = Image.open(io.BytesIO(convert_to_png(image_url)))
    # vis_processor converts to a torch tensor
    # img should have 4 dimensions
    img = vis_processor(img)
    img = torch.unsqueeze(img, 0)
    
    texts = prepare_texts([prompt], conv_temp)
    answers = model.generate(img,
                             texts,
                             max_new_tokens=500,
                             do_sample=True)
    print(answers)