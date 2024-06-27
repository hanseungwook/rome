import os

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin


def is_main_process():
    if not dist.is_initialized():
        if 'RANK' in os.environ:
            return int(os.environ['RANK']) == 0
        else:
            return True
    else:
        return dist.get_rank() == 0


def gather_tensors_in_dict(tensor_dict):
    world_size = dist.get_world_size()
    gathered_dict = {}
    for key, value in tensor_dict.items():
        gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
        dist.all_gather(gathered_values, value)
        gathered_dict[key] = torch.tensor(gathered_values)
        
    return gathered_dict


def setup_distributed_training(model, tok, train_loader, eval_loader, hparams):

    # if gradient accumulation > 1, enable it with plugin
    ga_plugin = None
    if hparams.gradient_accumulation_steps > 1:
        ga_plugin = GradientAccumulationPlugin(num_steps=hparams.gradient_accumulation_steps, sync_each_batch=True) # sync recommended for fsdp

    # Create an instance of the Accelerator class
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_plugin=ga_plugin)

    # Set the device of the model and tokenizer
    model, tok, train_loader, eval_loader = accelerator.prepare(model, tok, train_loader, eval_loader)

    return accelerator, model, tok, train_loader, eval_loader