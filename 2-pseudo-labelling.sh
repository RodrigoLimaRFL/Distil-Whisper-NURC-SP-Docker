# Name of the dataset in the Hugging Face Hub
_DATASET_NAME="RodrigoLimaRFL/nurc-sp-hugging-face"
# Split names to use for training, validation, and testing. Leave as default if you have all three splits.
_SPLITS="train+validation+test"
# Name of the text column in the dataset
_TEXT_COLUMN="text"
# Name of the new dataset to add to the Hugging Face Hub
_OUTPUT_DIR="nurc-sp_pseudo_labelled"
# Name of the project in the wandb hub
_WANDB_PROJECT="nurc-sp-distil-whisper-labelling"

echo pseudo-labelling &&
accelerate distil-whisper/training/run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "$_DATASET_NAME" \
  --dataset_split_name "$_SPLITS" \
  --text_column_name "$_TEXT_COLUMN" \
  --id_column_name "file_path" \
  --output_dir "$_OUTPUT_DIR" \
  --wandb_project "$_WANDB_PROJECT" \
  --per_device_eval_batch_size 8 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --concatenate_audio=True \
  --preprocessing_batch_size 125 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 2 \
  --report_to "wandb" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --push_to_hub
