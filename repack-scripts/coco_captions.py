from pathlib import Path
import json

from minigpt4.common.eval_utils import eval_parser
from minigpt4.common.config import Config

from MiniGPT4Captioner import MiniGPT4Captioner

if __name__ == '__main__':
    '''
    This repo only has minigpt for llama - go to the LURE repo for vicuna

    python /home/ngtj/MiniGPT-4/test_minigpt.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0
    '''
    parser = eval_parser()
    args = parser.parse_args()
    captioner = MiniGPT4Captioner()
    cfg = Config(args)

    # read dataset
    dataset_file = Path('../data/MSCOCO/coco-500-val2017_orig.json').absolute()
    print(f"Reading dataset {dataset_file}...", flush=True)

    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print("Success!", flush=True)

    # make all the captions
    captioner.generate_dataset(
        args,
        dataset=dataset,
        save_folder='/scratch/tim_kapil/MiniGPT-4',
        file_prefix='coco-500-val2017-minigpt4_llama',
        model_key='minigpt_vicuna',
        temp=1.0,
        top_p=0.95,
        half=None, # just doing the whole thing at once - not in as much of a rush
    )