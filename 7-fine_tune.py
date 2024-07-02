from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers import WhisperForConditionalGeneration
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Change variables to suit your needs

# Dataset link
dataset_link = "RodrigoLimaRFL/nurc-sp-hugging-face"

# Distilled model link
distilled_model_link = "RodrigoLimaRFL/distil-large-nurc-sp"

# Fine-tuned model name
fine_tuned_model_name = "distil-whisper-nurc-sp-fine-tuned"

# kwargs to define the model
kwargs = {
    "dataset_tags": "RodrigoLimaRFL/NURC-SP",
    "dataset": "NURC-SP",  # a 'pretty' name for the training dataset
    "dataset_args": "split: test",
    "language": "pt",
    "model_name": "NURC-SP distil-whisper fine-tuned",  # a 'pretty' name for your model
    "finetuned_from": "RodrigoLimaRFL/distil-large-nurc-sp",
    "tasks": "automatic-speech-recognition",
}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


nurc_sp = DatasetDict()

nurc_sp["train"] = load_dataset(
    dataset_link, split="train+validation"
)

nurc_sp["test"] = load_dataset(
    dataset_link, split="test"
)

print(nurc_sp)

nurc_sp = nurc_sp.select_columns(["audio", "text"])

feature_extractor = WhisperFeatureExtractor.from_pretrained(distilled_model_link)

tokenizer = WhisperTokenizer.from_pretrained(distilled_model_link, language="Portuguese", task="transcribe")

input_str = nurc_sp["train"][0]["text"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

processor = WhisperProcessor.from_pretrained(distilled_model_link, language="Portuguese", task="transcribe")

nurc_sp = nurc_sp.cast_column("audio", Audio(sampling_rate=16000))

print(nurc_sp["train"][0])

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

nurc_sp = nurc_sp.map(prepare_dataset, remove_columns=nurc_sp.column_names["train"], num_proc=1)

model = WhisperForConditionalGeneration.from_pretrained(distilled_model_link)

model.generation_config.language = "portuguese"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./" + fine_tuned_model_name,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=nurc_sp["train"],
    eval_dataset=nurc_sp["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

trainer.push_to_hub(**kwargs)