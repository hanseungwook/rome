from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    num_steps: int
    epochs: int
    eval_int: int
    save_int: int

    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    seed: int

    # Defaults
    use_ref: bool = False
    batch_size: int = 128
    gradient_accumulation_steps: int = 1

