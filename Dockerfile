FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN apt-get update && apt-get install -y \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y python3-pip \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH} \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN git clone https://github.com/huggingface/distil-whisper

RUN pip install --upgrade pip

RUN pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip uninstall transformer-engine -y;

RUN pip install -e distil-whisper/training

RUN pip uninstall flash-attn -y

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN apt-get install git-lfs

COPY 1-credentials.sh /workspace/1-credentials.sh

COPY 2-pseudo-labelling.sh /workspace/2-pseudo-labelling.sh

COPY 3-prepare-training.sh /workspace/3-prepare-training.sh

COPY 4-training.sh /workspace/4-training.sh

COPY 5-eval-nurc.sh /workspace/5-eval-nurc.sh

COPY 6-eval-whisper.sh /workspace/6-eval-whisper.sh

COPY 7-fine_tune.py /workspace/7-fine_tune.py

COPY 8-eval-fine-tuning.sh /workspace/8-eval-fine-tuning.sh

COPY commands.sh /workspace/commands.sh

RUN ["chmod", "+x", "/workspace/commands.sh"]

CMD ["/workspace/commands.sh"]