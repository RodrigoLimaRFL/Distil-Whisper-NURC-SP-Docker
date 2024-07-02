_HUGGING_FACE_TOKEN=""
_WANDB_TOKEN=""

echo credentials &&
export TOKENIZERS_PARALLERISM=true &&
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn && 
git config --global credential.helper store && 
huggingface-cli login --token "$_HUGGING_FACE_TOKEN" --add-to-git-credential && 
accelerate config default --mixed_precision bf16 && 
wandb login "$_WANDB_TOKEN" && 
huggingface-cli whoami && 
python3 /workspace/verify.py;