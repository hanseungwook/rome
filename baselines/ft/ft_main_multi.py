from copy import deepcopy
from typing import Any, Dict, List, Tuple

from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import ModelWithRef

from .ft_hparams import FTHyperParams


def apply_ft_to_model(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    opt: torch.optim.Optimizer,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    if copy:
        model = deepcopy(model)

    log_dict = execute_ft(accelerator, model, opt, tok, requests, hparams)

    return log_dict


def execute_ft(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    opt: torch.optim.Optimizer,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Boolean whether to use a reference model
    use_ref = False
    if isinstance(model.module, ModelWithRef):
        use_ref = True

    # Update target and print info
    requests = deepcopy(requests)
    requests['target_new']['str'] = [" " + r for r in requests['target_new']['str'] if r[0] != " "]

    # DEBUG prints
    # print('Executing algo for: ')
    for i in range(len(requests['prompt'])):
        print(f"[{requests['prompt'][i].format(requests['subject'][i])}] -> [{requests['target_new']['str'][i]}]\n")

    # Define inputs
    texts = [prompt.format(subject) for prompt, subject in zip(requests['prompt'], requests['subject'])]
    targets = requests['target_new']['str']
    txt, tgt = texts, targets

    prompt_inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
    prompt_lengths = prompt_inputs['attention_mask'].shape[1]

    if tok.add_bos_token:
        chosen_tgt = [tok.encode(t, padding=False)[1:] + [tok.eos_token_id] for t in tgt]
    else:
        chosen_tgt = [tok.encode(t, padding=False) + [tok.eos_token_id] for t in tgt]
    chosen_tgt = [torch.tensor(t) for t in chosen_tgt]
    chosen_tgt = pad_sequence(chosen_tgt, batch_first=True, padding_value=tok.pad_token_id)

    # unwrap model and generate from it
    with torch.no_grad():
        with FSDP.summon_full_params(model, recurse=False, writeback=False):
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.eval()
            model.model.eval() if use_ref else model.eval()
            # nucleus sampling -- gen = (bs * num_negatives, seq_len)
            gen = model.generate(input_ids=prompt_inputs['input_ids'], pad_token_id=tok.pad_token_id, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=32, max_new_tokens=20)
            model.model.train() if use_ref else model.train()

    # extract only responses (excluding prompt) and convert to tuple (for unique hashing)
    gen_ids = [tuple(o[prompt_inputs['input_ids'].shape[1]:].tolist()) for o in gen]
    gen_txt = tok.batch_decode(gen_ids, skip_special_tokens=True)

    for i in range(len(gen_txt)):
        print(f'PROMPT: {txt[i//32]}\t CHOSEN: {tgt[i//32]}\t GENERATED: {gen_txt[i]}\n')

    # create attention mask for chosen targets
    eos_token_idxs = ((chosen_tgt == tok.eos_token_id).cumsum(dim=1).cumsum(dim=1) == 1).argsort(dim=1)[:, -1]
    mask = torch.arange(chosen_tgt.shape[1]).expand(chosen_tgt.shape[0], chosen_tgt.shape[1]) <= eos_token_idxs.unsqueeze(1)
    chosen_attention_mask = torch.ones_like(chosen_tgt, dtype=torch.long) * mask

    chosen_tgt = {
        'input_ids': chosen_tgt.to('cuda'),
        'attention_mask': chosen_attention_mask.to('cuda'),
    }

    # concatenate prompt with target (eos token already added?)
    chosen_inputs = {
        'input_ids': torch.cat([prompt_inputs['input_ids'], chosen_tgt['input_ids']], dim=1),
        'attention_mask': torch.cat([prompt_inputs['attention_mask'], chosen_tgt['attention_mask']], dim=1)
    }

    # create labels for getting logprob by putting -100 for prompt and padding
    chosen_labels = chosen_inputs['input_ids'].clone()
    chosen_labels = chosen_labels.masked_fill(~chosen_inputs['attention_mask'].bool(), -100)
    chosen_labels[:, :prompt_lengths] = -100
    # shift by 1
    chosen_labels = chosen_labels[:, 1:].clone()

    current_chosen_logits = model(**chosen_inputs).logits
    current_chosen_logits = current_chosen_logits[:, :-1] # excluding last token logits

    chosen_logp_mask = chosen_labels != -100
    chosen_labels[chosen_labels == -100] = 0 # dummy tokens
    current_chosen_logp = torch.gather(F.log_softmax(current_chosen_logits, dim=-1), 2, chosen_labels.unsqueeze(2)).squeeze(2)
    current_chosen_logp = (current_chosen_logp * chosen_logp_mask).sum(-1) / chosen_logp_mask.sum(-1) # average over tokens
    loss = -current_chosen_logp.mean()

    accelerator.backward(loss)
    opt.step()
    opt.zero_grad()

    return {'loss': loss}


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk