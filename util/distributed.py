import torch
import torch.distributed as dist
from accelerate import Accelerator


def gather_tensors_in_dict(tensor_dict):
    world_size = dist.get_world_size()
    gathered_dict = {}
    for key, value in tensor_dict.items():
        gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
        dist.all_gather(gathered_values, value)
        gathered_dict[key] = gathered_values
        
    return gathered_dict


def setup_distributed_training(model, opt, tok, train_loader, mixed_precision):
    # Initialize the distributed backend
    # dist.init_process_group(backend='nccl')

    # Create an instance of the Accelerator class
    accelerator = Accelerator(mixed_precision=mixed_precision, log_with="wandb")

    # Set the device of the model and tokenizer
    model, opt, tok, train_loader = accelerator.prepare(model, opt, tok, train_loader)

    return accelerator, model, opt, tok, train_loader