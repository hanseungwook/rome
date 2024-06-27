export PYTHONPATH="/home/lab/rome/:/home/lab/rome/"
export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM="false"
set -e NCCL_BUFFSIZE

python -m accelerate.commands.launch --config_file ./accelerate_configs/fsdp_mistral7b.yaml ./experiments/evaluate_multi.py --alg_name=DPO --model_name=mistralai/Mistral-7B-v0.1 --hparams_fname=mistral7b_multi.json
