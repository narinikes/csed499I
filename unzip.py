import h5py
import numpy as np
import scipy.io.wavfile
import os
from tqdm import tqdm

# load all file name from directory
source_paths = ["./dataset/origin_data/train/tpa"]
target_paths = ["./dataset/unzipped/train/tpa"]


def to_audio(source, target):
    file_list = os.listdir(source)
    for file in tqdm(file_list):
        if file.endswith(".hd5"):
            # make path
            data = h5py.File(os.path.join(source, file), 'r')
            groups = list(data.keys())
            for group in groups:
                info = group.split()
                # pdb.set_trace()
                # info[0] = info[0][:3]
                # info[2] = 'Did'
                # info[4] = 'Tid'
                audio_name = info[1] + '.wav'
                audio_pcm = np.array(data[group]['audio'])
                scipy.io.wavfile.write(os.path.join(
                    target, audio_name), 16000, audio_pcm)


for source, target in zip(source_paths, target_paths):
    os.makedirs(target, exist_ok=True)
    to_audio(source, target)
