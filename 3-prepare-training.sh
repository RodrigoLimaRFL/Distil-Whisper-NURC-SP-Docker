# Distilled model link in the HuggingFace Hub (needs to be created beforehand)
_DISTILLED_MODEL_LINK="RodrigoLimaRFL/distil-large-nurc-sp"
# Name of the distilled model (needs to be the same as the hub)
_DISTILLED_MODEL_NAME="distil-large-nurc-sp"

echo prepare-training && 
cd /workspace && 
rm -rf distil-large-nurc-sp; 
git clone "$DISTILLED_MODEL" && 
cd "$_DISTILLED_MODEL_NAME" &&
cp ../distil-whisper/training/create_student_model.py . && 
cp ../distil-whisper/training/run_distillation.py . && 
python create_student_model.py \
    --teacher_checkpoint openai/whisper-large-v3 \
    --encoder_layers 32 \
    --decoder_layers 2 \
    --save_dir ./distil-large-v3-init