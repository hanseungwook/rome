import json
import shutil
from typing import Tuple, Union
from itertools import chain
from dataclasses import asdict

import torch
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
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    mixed_precision: str,
    dir_name: str,
):

    # Check if the environment variable exists
    if 'NCCL_BUFFSIZE' in os.environ:
        # Unset (delete) the environment variable
        del os.environ['NCCL_BUFFSIZE']
        print(f"Environment variable NCCL_BUFFSIZE has been unset.")

    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        checkpoint_dir = CHECKPOINT_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        if is_main_process():
            alg_dir = RESULTS_DIR / dir_name
            if alg_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1])
                    for x in alg_dir.iterdir()
                    if str(x).split("_")[-1].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
            checkpoint_dir = CHECKPOINT_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Results will be stored at {run_dir}") if is_main_process() else None
            print(f"Checkpoints will be stored at {checkpoint_dir}") if is_main_process() else None
        else:
            run_dir = None
            checkpoint_dir = None

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)

    # Set seed
    set_seed(hparams.seed)
    
    if is_main_process():
        if not (run_dir / "params.json").exists():
            shutil.copyfile(params_path, run_dir / "params.json")

    print(f"Executing {alg_name} with parameters {hparams}") if is_main_process() else None

    # Instantiate vanilla model
    print("Instantiating model") if is_main_process else None
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)

        if tok.pad_token is None or tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            
        tok.padding_side = "left" # for batch processing
    else:
        model, tok = model_name

    # if use_ref is True, then load reference model and wrap both models in 1 module
    ref_model = None
    if hasattr(hparams, 'use_ref') and hparams.use_ref is True:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model.eval()
        model = ModelWithRef(model, ref_model)

    if isinstance(model, ModelWithRef):
        for w in model.model.parameters():
            w.requires_grad = True
        for w in model.ref_model.parameters():
            w.requires_grad = False
    else:
        for w in model.parameters():
            w.requires_grad = True
    
    # Weights that are being optimized
    weights = [k for k, v in model.named_parameters() if v.requires_grad]
    print(f"Weights to be updated: {weights}") if is_main_process() else None

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data") if is_main_process() else None
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(ds, batch_size=hparams.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    # Call the setup_distributed_training function in the main function
    accelerator, model, tok, train_loader, eval_loader = setup_distributed_training(model, tok, train_loader, eval_loader, hparams)

    # Cast native autocast to generate function if using ModelWithRef
    if isinstance(model.module, ModelWithRef) and accelerator.mixed_precision in ('fp16', 'bf16'):
        model.module.generate = cast_with_native_amp(model.module.generate, accelerator.mixed_precision)
        model.model.generate = cast_with_native_amp(model.model.generate, accelerator.mixed_precision)
    else:
        model.generate = cast_with_native_amp(model.generate, accelerator.mixed_precision)
        
    # creating optimizer after setting up model b/c FSDP requires/recommends that
    opt = torch.optim.AdamW(
        model.model.parameters() if isinstance(model, ModelWithRef) else model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    opt = accelerator.prepare(opt)

    # Sync run_dir name across processes
    sync_run_dir = [run_dir, checkpoint_dir]
    accelerate.utils.broadcast_object_list(sync_run_dir, from_process=0)
    run_dir = sync_run_dir[0]
    checkpoint_dir = sync_run_dir[1]
    print(f'Synced dir names across processes: run_dir {run_dir}\tcheckpoint_dir {checkpoint_dir} ') if is_main_process() else None

    # Set up logging
    if accelerator.is_local_main_process:
        accelerator.init_trackers(
            project_name="knowledge-injection",
            config=asdict(hparams),
            init_kwargs={"wandb": {"entity": "llm-alignment"}}
        )

    meters = None
    if alg_name in ['DPO']:
        meters = {'loss': AverageMeter(), 'reward_acc': AverageMeter(), 'chosen_reward': AverageMeter(), 'rejected_reward': AverageMeter()}
    elif alg_name in ['FT']:
        meters = {'loss': AverageMeter}
    else:
        raise ValueError(f"Algorithm metrics {alg_name} not supported or implemented.")

    step = 0

    ##### Training loop #####
    for e in range(hparams.epochs):
        model.train()

        # Iterate through dataset in batches
        with tqdm(total=len(train_loader), desc=f"Training Epoch {e+1}/{hparams.epochs}", disable=not accelerator.is_local_main_process) as pbar:
            for batch in train_loader:
                # Compute weight changes + record weights that changed
                args_conserve_memory = (
                    dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                    if conserve_memory
                    else dict()
                )
                # with accelerator.accumulate(model):
                # requested_rewrites = [record["requested_rewrite"] for record in batch]
                log_dict = apply_algo(
                    accelerator,
                    model,
                    opt,
                    tok,
                    batch['requested_rewrite'],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
                )

                # gather tensors and log
                gathered_log_dict = gather_tensors_in_dict(log_dict)
                
                if accelerator.is_local_main_process:
                    for k, v in gathered_log_dict.items():
                        accelerator.log({k: v.mean().item()}, step=step)
                    
                    # update meters
                    for k, meter in meters.items():
                        if k in gathered_log_dict:
                            meter.update(gathered_log_dict[k].mean().item())

                    pbar.update(1)    
                    pbar.set_postfix({f'running {k}': v.avg for k, v in meters.items()})
                    
                    step += 1

        ##### Evaluation loop #####
        if (e + 1) % hparams.eval_int == 0:
            model.eval()
            # ref_model.eval()
            cur_run_dir = run_dir / f"epoch_{e}"

            # set up current run dir for the epoch
            if accelerator.is_local_main_process:    
                cur_run_dir.mkdir(parents=True, exist_ok=True)

            # assumes batch size of 1 (per process)
            with tqdm(total=len(eval_loader), desc=f"Evaluation Epoch {e+1}/{hparams.epochs}", disable=not accelerator.is_local_main_process) as pbar:
                for batch in eval_loader:
                    eval_results = []

                    batch['case_id'] = batch['case_id'].item()
                    for k in batch['requested_rewrite'].keys():

                        if isinstance(batch['requested_rewrite'][k], list):
                            batch['requested_rewrite'][k] = batch['requested_rewrite'][k][0]
                        else:
                            assert k in ['target_new', 'target_true']
                            for k1 in batch['requested_rewrite'][k].keys():
                                batch['requested_rewrite'][k][k1] = batch['requested_rewrite'][k][k1][0]

                    batch['paraphrase_prompts'] = list(chain(*batch['paraphrase_prompts']))
                    batch['neighborhood_prompts'] = list(chain(*batch['neighborhood_prompts']))
                    batch['attribute_prompts'] = list(chain(*batch['attribute_prompts']))
                    batch['generation_prompts'] = list(chain(*batch['generation_prompts']))

                    case_id = batch["case_id"]
                    case_result_path = cur_run_dir / f"case_{case_id}.json"

                    metrics = {
                        "case_id": case_id,
                        "requested_rewrite": batch["requested_rewrite"],
                        "post": ds_eval_method(model, tok, batch, snips, vec),
                    }
                    # metrics["pre"] = ds_eval_method(ref_model, tok, deepcopy(batch), snips, vec)
                    
                    eval_results.append(metrics)
                    gathered_eval_results = accelerator.gather_for_metrics(eval_results, use_gather_object=True)
                    if accelerator.is_local_main_process:
                        for metrics in gathered_eval_results:
                            if isinstance(metrics['case_id'], torch.Tensor):
                                metrics['case_id'] = metrics['case_id'].item()
                            case_id = metrics["case_id"]
                            case_result_path = cur_run_dir / f"case_{case_id}.json"
                            with open(case_result_path, "w") as f:
                                json.dump(metrics, f, indent=1)
                    pbar.update(1)

        ##### Save checkpoint #####
        if (e + 1) % hparams.save_int == 0:
            # save checkpoint
            print('Saving checkpoint...')
            accelerator.save_state(checkpoint_dir / f"epoch_{e+1}")

    accelerator.end_training()


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "DPO", "KN", "MEND", "KE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "mistralai/Mistral-7B-v0.1"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='no',
        help="Enable mixed precision training.",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        args.mixed_precision,
        dir_name=args.alg_name,
    )
