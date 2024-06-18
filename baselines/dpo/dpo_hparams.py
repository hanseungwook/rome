from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class DPOHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    epochs: int
    num_negatives: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    beta: float
    label_smoothing: float

    # Sampling parameters
    # num_samples: int
    # top_p: float
    # top_k: int
    # temperature: float
    # max_len: int

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Defaults
    batch_size: int = 128
