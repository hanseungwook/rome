from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class DPOHyperParams(HyperParams):
    # Method
    # layers: List[int]
    num_steps: int
    epochs: int
    eval_int: int
    save_int: int
    
    num_negatives: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    beta: float
    label_smoothing: float
    use_ref: bool

    seed: int

    # Defaults
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    
