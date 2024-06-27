from typing import Callable, Optional
from accelerate.utils import convert_outputs_to_fp32, is_torch_version

import torch


def cast_with_native_amp(func: Callable, mixed_precision: Optional[str] = None) -> Callable:
    """Almost like how huggingface accelerate cast `model.forward`."""
    if mixed_precision not in ("fp16", "bf16"):
        print(f"Unknown mixed precision mode: {mixed_precision}, falling back to fp32.")
        return func

    if mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
        output_func = torch.cuda.amp.autocast(dtype=torch.float16)(func)
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        output_func = torch.autocast(device_type=device_type, dtype=torch.bfloat16)(func)
    # output_func = convert_outputs_to_fp32(output_func)
    return output_func