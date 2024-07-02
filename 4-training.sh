# Absolute path of the directory where the model is stored
_DIR_PATH="/workspace/distil-large-nurc-sp"
# Name of the pseudo-labelled dataset in the Hugging Face Hub
_DATASET_NAME="RodrigoLimaRFL/nurc-sp_pseudo_labelled"
# Name of the text column in the dataset
_TEXT_COLUMN="text"

echo training && 
cd "$_DIR_PATH" &&
accelerate launch run_distillation.py   \
    --model_name_or_path ./distil-large-v3-init  \ 
    --teacher_model_name_or_path openai/whisper-large-v3   \
    --train_dataset_name "$DATASET_NAME"   \
    --train_split_name train   \
    --text_column_name "$TEXT_COLUMN"   \
    --train_dataset_samples 220   \
    --eval_dataset_name "$DATASET_NAME"   \
    --eval_split_name validation   \
    --eval_text_column_name "$TEXT_COLUMN"   \
    --eval_steps 1000   \
    --save_steps 5000   \
    --warmup_steps 50   \
    --learning_rate 0.0001   \
    --lr_scheduler_type constant_with_warmup   \
    --timestamp_probability 0.2   \
    --condition_on_prev_probability 0.2   \
    --language pt   \
    --task transcribe   \
    --logging_steps 25   \
    --save_total_limit 1   \
    --max_steps 50000   \
    --wer_threshold 20   \
    --per_device_train_batch_size 16   \
    --per_device_eval_batch_size 16   \
    --dataloader_num_workers 8   \
    --preprocessing_num_workers 8   \
    --ddp_timeout 7200   \
    --dtype bfloat16   \
    --attn_implementation sdpa   \
    --output_dir ./   \
    --do_train   \
    --do_eval      \
    --overwrite_output_dir  \ 
    --predict_with_generate   \
    --freeze_encoder   \
    --freeze_embed_positions  \ 
    --streaming False   \
    --push_to_hub
