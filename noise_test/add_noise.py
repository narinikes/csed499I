import os
import torchaudio
import torch

source_paths = ["./dataset/unzipped/train/tpa"]
target_paths = ["./dataset/noised/noise02"]
noise_path = "./dataset/noise/noise02.wav"
snr_lists = [0, 5, 10, 15, 20, 25]

def adding_noise(source, target, snr):
    file_list = os.listdir(source)
    for i in range(10):
        file  = file_list[i]
        if file.endswith(".wav"):
            waveform_data, sample_rate_data = torchaudio.load(os.path.join(source, file))
            waveform_noise, _ = torchaudio.load(noise_path)
            waveform_noise = waveform_noise[:, :waveform_data.shape[1]]
            noising = torchaudio.transforms.AddNoise()
            snr_list = torch.tensor([snr])
            noised_data = noising(waveform_data, waveform_noise, snr_list)
            torchaudio.save(os.path.join(target, str(snr), file), noised_data, sample_rate_data)

for source, target in zip(source_paths, target_paths):
    os.makedirs(target, exist_ok=True)
    for snr in snr_lists:
        os.makedirs(os.path.join(target, str(snr)), exist_ok=True)
        adding_noise(source, target, snr)