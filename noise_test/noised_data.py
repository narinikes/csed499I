#test with 10 noised01 data) wer: 1.1667, 1.1667, 1.1574, 1.1481, 1.1481, 1.1481
#test with 10 noise02 data) wer: 1.1389, 1.1574, 1.1574, 1.1574, 1.1574, 1.1389

from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import os
import jsonlines
from jiwer import wer

source = "./dataset/noised/noise02"
transcript = "./dataset/transcript/train.jsonl"
snr_lists = [0, 5, 10, 15, 20, 25]
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
trans = [[0], [0], [0], [0], [0], [0]]

#get scripts
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

#prediction
def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    trans[int(snr/5)].append(transcription[0])
    return batch

#check wer
def evaluate(snr, script, wer_list):
    eval_data = load_dataset(os.path.join(source, str(snr)))
    result = eval_data.map(map_to_pred, remove_columns=["audio"])
    w = wer(script, trans[int(snr/5)][1:])
    print("WER:", w)
    wer_list.append(w)


wer_list = []
for snr in snr_lists:
    
    script = read_script()
    evaluate(snr, script, wer_list)
    
print(wer_list)