HUGGING_FACE_TOKEN=""

sh /workspace/1-credentials.sh
sh /workspace/2-pseudo-labelling.sh
sh /workspace/3-prepare-training.sh
sh /workspace/4-training.sh
sh /workspace/5-eval-nurc.sh
sh /workspace/6-eval-whisper.sh
python3 /workspace/7-fine_tune.py
sh /workspace/8-eval-fine-tuning.sh