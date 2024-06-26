from copy import deepcopy
from typing import Any, Dict, List, Tuple
import re

from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from util import nethook

from .dpo_hparams import DPOHyperParams

def print_grad(opt):
    for group in opt.param_groups:
        for param in group['params']:
            print(f"Gradient: \n{param.grad}")

def apply_dpo_to_model(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    opt: torch.optim.Optimizer,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
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

    log_dict = execute_dpo(accelerator, model, ref_model, opt, tok, requests, hparams)

    return log_dict


def execute_dpo(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    opt: torch.optim.Optimizer,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DPOHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the DPO update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    requests['target_new']['str'] = [" " + r for r in requests['target_new']['str'] if r[0] != " "]
    print('Executing algo for: ')
    for i in range(len(requests['prompt'])):
        print(f"[{requests['prompt'][i].format(requests['subject'][i])}] -> [{requests['target_new']['str'][i]}]")

    # Define inputs
    texts = [prompt.format(subject) for prompt, subject in zip(requests['prompt'], requests['subject'])]
    targets = requests['target_new']['str']
    txt, tgt = texts, targets

    chunk_bs = len(txt)
    prompt_inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
    # chosen_tgt = tok(tgt, return_tensors="pt", padding=True).to("cuda")

    chosen_tgt = [tok.encode(t, padding=False) + [tok.eos_token_id] for t in tgt]
    chosen_tgt = [torch.tensor(t) for t in chosen_tgt]
    chosen_tgt = pad_sequence(chosen_tgt, batch_first=True, padding_value=tok.pad_token_id)

    # create attention mask for generated targets
    eos_token_idxs = ((chosen_tgt == tok.eos_token_id).cumsum(dim=1).cumsum(dim=1) == 1).argsort(dim=1)[:, -1]
    mask = torch.arange(chosen_tgt.shape[1]).expand(chosen_tgt.shape[0], chosen_tgt.shape[1]) <= eos_token_idxs.unsqueeze(1)
    chosen_attention_mask = torch.ones_like(chosen_tgt, dtype=torch.long) * mask

    chosen_tgt = {
        'input_ids': chosen_tgt.to('cuda'),
        'attention_mask': chosen_attention_mask.to('cuda'),
    }
    
    # unwrap model and generate from it
    with torch.no_grad() and FSDP.summon_full_params(model, recurse=False, writeback=False):
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        # nucleus sampling -- gen = (bs * num_negatives, seq_len)
        gen = unwrapped_model.generate(input_ids=prompt_inputs['input_ids'], do_sample=True, top_p=0.95, top_k=50, num_return_sequences=hparams.num_negatives, max_new_tokens=20)
        unwrapped_model.train()
        model.train()
    
    # extract only responses (excluding prompt) and convert to tuple (for unique hashing)
    gen_ids = [tuple(o[prompt_inputs['input_ids'].shape[1]:].tolist()) for o in gen]
    gen_txt = tok.batch_decode(gen_ids, skip_special_tokens=True)
    
    def truncate_at_first_punctuation(s):
        # Define a regular expression pattern to match any punctuation
        trunc_pattern = re.compile(r'[.!?;:\n]')
        match = trunc_pattern.search(s)
        
        # If punctuation is found, truncate the string at that position
        if match:
            result = s[:match.start()+1]
        else:
            result = s  # No punctuation found, return the original string
        
        return result


    trunc_gen_txt = list(map(truncate_at_first_punctuation, gen_txt))
    # tokenize this truncated text, add eos token, and pad to longest length
    gen_tgt = [tok.encode(t, padding=False) + [tok.eos_token_id] for t in trunc_gen_txt]
    gen_tgt = [torch.tensor(t) for t in gen_tgt]
    gen_tgt = pad_sequence(gen_tgt, batch_first=True, padding_value=tok.pad_token_id)
    
    # for i in range(len(trunc_gen_txt)):
    #     print('PROMPT: ', txt[i//hparams.num_negatives])
    #     print('CHOSEN: ', tgt[i//hparams.num_negatives])
    #     print('GENERATED: ', trunc_gen_txt[i])

    # create attention mask for generated targets
    eos_token_idxs = ((gen_tgt == tok.eos_token_id).cumsum(dim=1).cumsum(dim=1) == 1).argsort(dim=1)[:, -1]
    mask = torch.arange(gen_tgt.shape[1]).expand(gen_tgt.shape[0], gen_tgt.shape[1]) <= eos_token_idxs.unsqueeze(1)
    gen_attention_mask = torch.ones_like(gen_tgt, dtype=torch.long) * mask

    rejected_tgt = {
        'input_ids': gen_tgt.to('cuda'),
        'attention_mask': gen_attention_mask.to('cuda'),
    }
    
    bs = gen_tgt.shape[0]

    # concatenate prompt with target (eos token already added)
    chosen_inputs = {
        'input_ids': torch.cat([prompt_inputs['input_ids'], chosen_tgt['input_ids']], dim=1),
        'attention_mask': torch.cat([prompt_inputs['attention_mask'], chosen_tgt['attention_mask']], dim=1)
    }
    rejected_inputs = {
        'input_ids': torch.cat([prompt_inputs['input_ids'].repeat_interleave(hparams.num_negatives, dim=0), rejected_tgt['input_ids']], dim=1),
        'attention_mask': torch.cat([prompt_inputs['attention_mask'].repeat_interleave(hparams.num_negatives, dim=0), rejected_tgt['attention_mask']], dim=1)
    }

    prompt_lengths = prompt_inputs['attention_mask'].shape[1] # constant prompt length (because left-padded)

    # create labels for getting logprob
    chosen_labels = chosen_inputs['input_ids'].clone()
    # put -100 for prompt and padding
    chosen_labels = chosen_labels.masked_fill(~chosen_inputs['attention_mask'].bool(), -100)
    chosen_labels[:, :prompt_lengths] = -100

    rejected_labels = rejected_inputs['input_ids'].clone()
    rejected_labels = rejected_labels.masked_fill(~rejected_inputs['attention_mask'].bool(), -100)
    rejected_labels[:, :prompt_lengths] = -100

    # get logits for chosen and rejected under current model
    current_chosen_logits = model(**chosen_inputs).logits
    current_rejected_logits = model(**rejected_inputs).logits
    # with torch.no_grad() and FSDP.summon_full_params(ref_model, recurse=False, writeback=False):
        # ref_chosen_logits = ref_model(**chosen_inputs).logits
        # ref_rejected_logits = ref_model(**rejected_inputs).logits

    # TODO: do this filtering out at the loss level (where it's 0 b/c chosen == generated)
    # get logp with the chosen and rejected labels
    chosen_logp_mask = chosen_labels != -100
    rejected_logp_mask = rejected_labels != -100
    chosen_labels[chosen_labels == -100] = 0 # dummy tokens
    rejected_labels[rejected_labels == -100] = 0

    current_chosen_logp = torch.gather(F.log_softmax(current_chosen_logits, dim=-1), 2, chosen_labels.unsqueeze(2)).squeeze(2)
    current_chosen_logp = (current_chosen_logp * chosen_logp_mask).sum(-1)
    current_rejected_logp = torch.gather(F.log_softmax(current_rejected_logits, dim=-1), 2, rejected_labels.unsqueeze(2)).squeeze(2)
    current_rejected_logp = (current_rejected_logp * rejected_logp_mask).sum(-1)
    # ref_chosen_logp = torch.gather(F.log_softmax(ref_chosen_logits, dim=-1), 2, chosen_labels.unsqueeze(2)).squeeze(2)
    # ref_chosen_logp = (ref_chosen_logp * chosen_logp_mask).sum(-1)
    # ref_rejected_logp = torch.gather(F.log_softmax(ref_rejected_logits, dim=-1), 2, rejected_labels.unsqueeze(2)).squeeze(2)
    # ref_rejected_logp = (ref_rejected_logp * rejected_logp_mask).sum(-1)

    current_logratios = current_chosen_logp.unsqueeze(1) - current_rejected_logp.reshape(-1, hparams.num_negatives)
    # ref_logratios = ref_chosen_logp.unsqueeze(1) - ref_rejected_logp.reshape(-1, hparams.num_negatives)

    if ref_model is None:
        ref_logratios = 0.
        ref_chosen_logp = 0.
        ref_rejected_logp = 0.
    
    logits = current_logratios - ref_logratios
    # label_smoothing = 0 gives original DPO
    # TODO: maybe implement length regularization here, maybe mask out zeros (chosen == rejected) in logratios
    loss = -F.logsigmoid(hparams.beta * logits) * (1 - hparams.label_smoothing) - F.logsigmoid(-hparams.beta * logits) * hparams.label_smoothing
    loss = loss.mean()
    chosen_rewards = hparams.beta * (current_chosen_logp - ref_chosen_logp).detach().repeat_interleave(hparams.num_negatives, dim=0)
    rejected_rewards = hparams.beta * (current_rejected_logp - ref_rejected_logp).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()

    opt.zero_grad()
    accelerator.backward(loss)
    opt.step()

    # TODO: skipping norm constraint 
    # if type(hparams.norm_constraint) is float:
    #     eps = hparams.norm_constraint
    #     with torch.no_grad():
    #         for k, v in weights.items():
    #             v[...] = torch.clamp(
    #                 v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
    #             )

    return {'loss': loss, 
            'reward_acc': reward_accuracies.mean(),
            'chosen_reward': chosen_rewards.mean(),
            'rejected_reward': rejected_rewards.mean(),
            }


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