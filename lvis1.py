from minigpt4.common.eval_utils import eval_parser
from minigpt4.common.config import Config
import json

from generate_hat_captions import MiniGPT4Captioner

if __name__ == "__main__":
    parser = eval_parser()
    args = parser.parse_args()
    captioner = MiniGPT4Captioner()
    cfg = Config(args)

    # read HAT
    print("Reading LVIS dataset...", flush=True)
    lvis_file = './data/LVIS/lvis_1000_empty.json'

    with open(lvis_file, 'r', encoding='utf-8') as f:
        lvis_data = json.load(f)

    print("Success!", flush=True)

    # make all the captions
    captioner.generate_dataset(
        args,
        lvis_data,
        file_prefix="lvis_minigpt",
        model_key='gptmini',
        half=1
    )