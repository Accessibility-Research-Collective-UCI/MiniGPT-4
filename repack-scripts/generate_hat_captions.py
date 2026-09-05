import json

from minigpt4.common.eval_utils import eval_parser
from minigpt4.common.config import Config

from MiniGPT4Captioner import MiniGPT4Captioner


if __name__ == "__main__":
    '''
    python /home/ngtj/MiniGPT-4/test_minigpt.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
    '''
    parser = eval_parser()
    args = parser.parse_args()
    captioner = MiniGPT4Captioner()
    cfg = Config(args)

    # read HAT
    print("Reading HAT dataset...", flush=True)
    hat_file = './data/hat_empty_prepared.json'

    with open(hat_file, 'r', encoding='utf-8') as f:
        hat_data = json.load(f)

    print("Success!", flush=True)

    # make all the captions
    captioner.generate_dataset(
        args,
        hat_data,
        half=1
    )