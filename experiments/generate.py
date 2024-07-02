import json
import shutil
from typing import Tuple, Union
from itertools import chain
from dataclasses import asdict
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import accelerate
from accelerate.utils import set_seed

# from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.dpo import DPOHyperParams, apply_dpo_to_model, ModelWithRef
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from util.metrics import AverageMeter
from util.distributed import *
from util.amp import cast_with_native_amp

use_flash_attn = False
try:
    import flash_attn
    use_flash_attn = True
    print('Enabling flash attention 2') if is_main_process() else None
except ImportError:
    pass

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "DPO": (DPOHyperParams, apply_dpo_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    # "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

def main(
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    checkpoint_path: str,
):

    # Check if the environment variable exists
    if 'NCCL_BUFFSIZE' in os.environ:
        # Unset (delete) the environment variable
        del os.environ['NCCL_BUFFSIZE']
        print(f"Environment variable NCCL_BUFFSIZE has been unset.")

    # Instantiate vanilla model
    print("Instantiating model...") if is_main_process else None
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)

        if tok.pad_token is None or tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            
        tok.padding_side = "left" # for batch processing
    else:
        model, tok = model_name

    # Load from checkpoint
    assert checkpoint_path is not None
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to("cuda")
    print("Loaded checkpoint...")

    # Load data
    print("Loading dataset...") if is_main_process() else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    
    # Create data loader
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    # Generate over the evaluation dataset
    with tqdm(total=len(eval_loader), desc="Generating") as pbar:
        for batch in eval_loader:
            requests = deepcopy(batch['requested_rewrite'])
            requests['target_new']['str'] = [" " + r for r in requests['target_new']['str'] if r[0] != " "]

            # Define inputs
            texts = [prompt.format(subject) for prompt, subject in zip(requests['prompt'], requests['subject'])]
            targets = requests['target_new']['str']
            txt, tgt = texts, targets

            prompt_inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
            model.eval()

            with torch.no_grad():
                gen = model.generate(input_ids=prompt_inputs['input_ids'], pad_token_id=tok.pad_token_id, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=16, max_new_tokens=20)

            # # extract only responses (excluding prompt) and convert to tuple (for unique hashing)
            gen_ids = [tuple(o[prompt_inputs['input_ids'].shape[1]:].tolist()) for o in gen]
            gen_txt = tok.batch_decode(gen_ids, skip_special_tokens=True)

            for i in range(len(gen_txt)):
                print(f'PROMPT: {txt[i]}\t CHOSEN: {tgt[i]}\t GENERATED: {gen_txt[i]}\n')

            pbar.update(1)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "mistralai/Mistral-7B-v0.1"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path to load from.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )

    # parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.model_name,
        args.ds_name,
        args.dataset_size_limit,
        args.checkpoint_path,
    )
