#test with 10 data) wer: 1.1481

from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import os
import jsonlines
from jiwer import wer

source = "./dataset/unzipped/train/tpa"
transcript = "./dataset/transcript/train.jsonl"
snr_lists = [0, 5, 10, 15, 20, 25]
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
trans = []

def read_script():
    script = []
    with jsonlines.open(transcript) as f:
        i = 0
        for line in f:
            if i >= 10:
                break
            script.append(line['script'])
            i += 1
    return script

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    trans.append(transcription[0])
    return batch

def evaluate(script):
    eval_data = load_dataset(source, split='train[:10]')
    result = eval_data.map(map_to_pred, remove_columns=["audio"])
    script = script[:len(trans)]
    w = wer(script, trans)
    print("WER:", w)
    
script = read_script()
evaluate(script)