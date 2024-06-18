from copy import deepcopy
from typing import Any, Dict, List, Tuple

from accelerate import Accelerator
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    execute_dpo(accelerator, ref_model, model, opt, tok, requests, hparams)

    return


def execute_dpo(
    accelerator: Accelerator,
    ref_model: AutoModelForCausalLM,
    model: AutoModelForCausalLM,
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
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]
    txt, tgt = texts, targets

    chunk_bs = len(txt)
    prompt_inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
    chosen_tgt = tok(tgt, return_tensors="pt", padding=True).to("cuda")

    # nucleus sampling
    gen = model.generate(input_ids=prompt_inputs['input_ids'], do_sample=True, top_p=0.95, top_k=50, num_return_sequences=hparams.num_negatives, max_length=10)
    gen = gen.reshape(-1, hparams.num_negatives, gen.shape[1]) # reshape into (bs, num_negatives, seq_len)
    # greedy decoding
    # gen = model.generate(input_ids=prompt_inputs['input_ids'], max_length=10)
    
    # extract only responses (excluding prompt) and convert to tuple (for unique hashing)
    gen_ids = [tuple(o[prompt_inputs['input_ids'].shape[1]:].tolist()) for o in gen]
    gen_txt = [tok.decode(o, skip_special_tokens=True) for o in gen_ids]
    print('PROMPT: ', txt)
    print('CHOSEN: ', tgt)
    print('GENERATED: ', gen_txt)

    # Use a set to filter out duplicates
    unique_gen_ids = list(set(gen_ids))

    # Convert the unique token ID tuples back to tensors
    gen_tgt = torch.stack([torch.tensor(o, dtype=torch.long) for o in unique_gen_ids], dim=0).to('cuda')

    # Filter out generations that are the same as the new target
    matches = [torch.equal(g, t) for g, t in zip(gen_tgt, chosen_tgt['input_ids'].repeat(gen_tgt.shape[0], 1))]
    gen_tgt = gen_tgt[~torch.tensor(matches)]
    if len(gen_tgt) == 0:
        print('No valid negative generations found!')
        continue

    rejected_tgt = {
        'input_ids': gen_tgt,
        'attention_mask': (gen_tgt != tok.eos_token_id).to('cuda') # pad_token_id == eos_token_id
    }

    # ensure eos token is not in prompt or target
    assert tok.eos_token_id not in prompt_inputs['input_ids']
    assert tok.eos_token_id not in chosen_tgt['input_ids']
    # assert tok.eos_token_id not in rejected_tgt['input_ids']
    
    # TODO: this will be a problem when prompt_inputs have different lengths and bs > 1
    
    bs = gen_tgt.shape[0]

    # concatenate prompt with target and add eos token
    chosen_inputs = {
        'input_ids': torch.cat([prompt_inputs['input_ids'], chosen_tgt['input_ids'], torch.full((chunk_bs, 1), tok.eos_token_id, device='cuda', dtype=torch.long)], dim=1),
        'attention_mask': torch.cat([prompt_inputs['attention_mask'], chosen_tgt['attention_mask'], torch.ones((chunk_bs, 1), device='cuda', dtype=torch.long)], dim=1)
    }
    rejected_inputs = {
        'input_ids': torch.cat([prompt_inputs['input_ids'].repeat(bs, 1), rejected_tgt['input_ids'], torch.full((bs, 1), tok.eos_token_id, device='cuda', dtype=torch.long)], dim=1),
        'attention_mask': torch.cat([prompt_inputs['attention_mask'].repeat(bs, 1), rejected_tgt['attention_mask'], torch.ones((bs, 1), device='cuda', dtype=torch.long)], dim=1)
    }

    prompt_lengths = prompt_inputs['attention_mask'].sum(dim=1)
    chosen_seq_len = chosen_inputs['attention_mask'].shape[1]
    rejected_seq_len = rejected_inputs['attention_mask'].shape[1]

    # mask for prompt
    chosen_prompt_mask = torch.arange(chosen_seq_len, device='cuda').expand(chunk_bs, chosen_seq_len) < prompt_lengths.unsqueeze(1)
    rejected_prompt_mask = torch.arange(rejected_seq_len, device='cuda').expand(chunk_bs, rejected_seq_len) < prompt_lengths.unsqueeze(1)

    # create labels for getting logprob
    chosen_labels = chosen_inputs['input_ids'].clone()
    chosen_labels = chosen_labels.masked_fill(chosen_prompt_mask, -100).repeat(bs, 1) # repeat along bs do we need this?
    rejected_labels = rejected_inputs['input_ids'].clone()
    rejected_labels = rejected_labels.masked_fill(rejected_prompt_mask, -100)

    # get logits for chosen and rejected under current model
    current_chosen_logits = model(**chosen_inputs).logits
    current_rejected_logits = model(**rejected_inputs).logits
    with torch.no_grad():
        ref_chosen_logits = ref_model(**chosen_inputs).logits
        ref_rejected_logits = ref_model(**rejected_inputs).logits

    # repeat chosen outputs to match batch size
    if bs > 1:
        current_chosen_logits = current_chosen_logits.repeat(bs, 1, 1)
        ref_chosen_logits = ref_chosen_logits.repeat(bs, 1, 1) # TODO: do we need this?

    # get logp with the chosen and rejected labels
    chosen_logp_mask = chosen_labels != -100
    rejected_logp_mask = rejected_labels != -100
    chosen_labels[chosen_labels == -100] = 0
    rejected_labels[rejected_labels == -100] = 0

    current_chosen_logp = torch.gather(F.log_softmax(current_chosen_logits, dim=-1), 2, chosen_labels.unsqueeze(2)).squeeze(2)
    current_chosen_logp = (current_chosen_logp * chosen_logp_mask).sum(-1)
    current_rejected_logp = torch.gather(F.log_softmax(current_rejected_logits, dim=-1), 2, rejected_labels.unsqueeze(2)).squeeze(2)
    current_rejected_logp = (current_rejected_logp * rejected_logp_mask).sum(-1)
    ref_chosen_logp = torch.gather(F.log_softmax(ref_chosen_logits, dim=-1), 2, chosen_labels.unsqueeze(2)).squeeze(2)
    ref_chosen_logp = (ref_chosen_logp * chosen_logp_mask).sum(-1)
    ref_rejected_logp = torch.gather(F.log_softmax(ref_rejected_logits, dim=-1), 2, rejected_labels.unsqueeze(2)).squeeze(2)
    ref_rejected_logp = (ref_rejected_logp * rejected_logp_mask).sum(-1)

    current_logratios = current_chosen_logp - current_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp

    logits = current_logratios - ref_logratios
    # label_smoothing = 0 gives original DPO
    loss = -F.logsigmoid(hparams.beta * logits) * (1 - hparams.label_smoothing) - F.logsigmoid(-hparams.beta * logits) * hparams.label_smoothing
    loss = loss.mean()
    chosen_rewards = hparams.beta * (current_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = hparams.beta * (current_rejected_logp - ref_rejected_logp).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()

    opt.zero_grad()
    
    # TODO: fix chunk_bs if you generate more than 1 response
    print(f"Batch loss {loss.item()}")
    print(f'Reward accuracy {reward_accuracies.mean().item()}')
    loss_meter.update(loss.item(), n=chunk_bs)
    acc_meter.update(reward_accuracies.mean().item(), n=chunk_bs)

    if loss.item() >= 1e-2:
        loss.backward()
        opt.step()

    if type(hparams.norm_constraint) is float:
        eps = hparams.norm_constraint
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = torch.clamp(
                    v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                )

return deltas


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


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count