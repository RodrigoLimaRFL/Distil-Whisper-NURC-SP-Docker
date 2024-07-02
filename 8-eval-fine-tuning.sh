# Absolute path of the directory where the model is stored
_DIR_PATH="/workspace/distil-large-nurc-sp"
# Name of the pseudo-labelled dataset in the Hugging Face Hub
_DATASET_NAME="RodrigoLimaRFL/nurc-sp_pseudo_labelled"
# Link to the fine-tuned model
_MODEL_NAME="RodrigoLimaRFL/distil-whisper-nurc-sp-fine-tuned"

echo eval-whisper && 
cd "$_DIR_PATH" && 
python run_eval.py   --model_name_or_path "$_MODEL_NAME"  --dataset_name "$_DATASET_NAME" --dataset_config_name "default"  --dataset_split_name test   --text_column_name text   --batch_size 16   --dtype bfloat16   --generation_max_length 256   --language pt   --attn_implementation sdpa
