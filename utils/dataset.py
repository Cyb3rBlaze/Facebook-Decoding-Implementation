import torch
import torchaudio
from torch.utils.data import Dataset

from tqdm import tqdm

import scipy.io
from scipy.signal import butter, sosfilt, freqs

import matplotlib.pyplot as plt

import numpy as np

import IPython

import os

import wave


# custom dataset used to load pairs for training
class CustomDataset(Dataset):
    def __init__(self, subject_path, audio_dir):
        mat = scipy.io.loadmat(subject_path)
        raw_data = mat["raw"][0][0][3][0][0]
        sos = butter(10, (0.1, 200), 'bandpass', fs=500, output='sos')
        filtered = sosfilt(sos, raw_data[0])

        brain_data = torch.tensor(filtered)

        # for resampling purposes
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        self.audio_sample_rate = bundle.sample_rate
        self.brain_data_sample_rate = 500


        all_audio_data = None

        for _, file in enumerate(os.listdir(audio_dir)):
            waveform, sample_rate = torchaudio.load(audio_dir + file)

            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
            
            if _ == 0:
                all_audio_data = waveform
            else:
                all_audio_data = torch.hstack((all_audio_data, waveform))


        # reshape for 3 second samples

        all_audio_data = all_audio_data[0][:all_audio_data.shape[1]//(self.audio_sample_rate*3)*(self.audio_sample_rate*3)]

        all_audio_data = all_audio_data.view(-1, self.audio_sample_rate*3)

        brain_data = brain_data[:, :brain_data.shape[1]//1500*1500].view(brain_data.shape[0], -1, 1500)

        print("Brain data shape: " + str(brain_data.shape))
        print("Waveform shape: " + str(all_audio_data.shape))
        print("Audio sampling rate (Hz): " + str(self.audio_sample_rate))
        print("Brain data sampling rate (Hz): 500")

        self.num_samples = min(brain_data.shape[1], all_audio_data.shape[0])

        # tranpose channel and num_samples dims
        self.brain_data = torch.transpose(brain_data[:, :self.num_samples], 0, 1)
        self.all_audio_data = all_audio_data[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.brain_data[idx], self.all_audio_data[idx])

    def get_audio_sample_rate(self):
        return self.audio_sample_rate

    def get_brain_data_sample_rate(self):
        return 500