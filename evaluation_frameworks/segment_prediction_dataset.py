import torch
import torchaudio
from torch.utils.data import Dataset

from tqdm import tqdm

import scipy.io
from scipy.signal import butter, sosfilt, resample

import matplotlib.pyplot as plt

import numpy as np

import IPython

import os

import wave

from tqdm import tqdm


# custom dataset used to load pairs for training
class CustomDataset(Dataset):
    def __init__(self, data_dir, T_out, num_subjects, exclude):

        all_subject_brain_data = None

        # for resampling purposes
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53

        self.audio_sample_rate = 16000
        self.brain_data_sample_rate = 150

        # 49 subjects
        for i in tqdm(range(1, num_subjects + 1)):
            if i not in exclude:
                if i < 10:
                    mat = scipy.io.loadmat(data_dir + "/S0" + str(i) + ".mat")
                else:
                    mat = scipy.io.loadmat(data_dir + "/S" + str(i) + ".mat")
                raw_data = mat["raw"][0][0][3][0][0]
                sos = butter(10, (0.1, 200), 'bandpass', fs=500, output='sos')

                brain_data = None

                for channel in raw_data:
                    final_filtered = sosfilt(sos, channel)

                    final_filtered = resample(final_filtered, int(len(final_filtered)*(self.brain_data_sample_rate/500)))

                    final_filtered = final_filtered - np.mean(final_filtered[:int(self.brain_data_sample_rate*0.5)])
                    final_filtered = (final_filtered - np.mean(final_filtered)) / np.std(final_filtered)
                    final_filtered = np.clip(final_filtered, -20, 20)

                    if brain_data == None:
                        brain_data = torch.tensor(final_filtered)
                    else:
                        brain_data = torch.vstack((brain_data, torch.tensor(final_filtered)))
                
                if i == 1:
                    all_subject_brain_data = torch.unsqueeze(brain_data, 0)
                else:
                    min_time_dim = min(brain_data.shape[-1], all_subject_brain_data.shape[-1])
                    all_subject_brain_data = torch.vstack((all_subject_brain_data[:, :61, :min_time_dim], torch.unsqueeze(brain_data, 0)[:, :61, :min_time_dim]))

        num_subjects = num_subjects - len(exclude)

        # audio file parsing
        all_audio_data = None

        for _, file in enumerate(os.listdir(data_dir + "/audio")):
            waveform, sample_rate = torchaudio.load(data_dir + "/audio/" + file)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.audio_sample_rate)
            
            if _ == 0:
                all_audio_data = waveform
            else:
                all_audio_data = torch.hstack((all_audio_data, waveform))

        # reshape for 3 second samples

        all_audio_data = all_audio_data[0][:all_audio_data.shape[1]//(self.audio_sample_rate*3)*(self.audio_sample_rate*3)]

        all_audio_data = all_audio_data.view(-1, self.audio_sample_rate*3).repeat(num_subjects, 1, 1).reshape((-1, 16000))

        self.all_subject_brain_data = torch.mean(all_subject_brain_data[:, :, :all_subject_brain_data.shape[-1]//450*450].view(num_subjects, all_subject_brain_data.shape[1], -1, 50, 9), dim=4)[:, :, :, :T_out]

        self.num_samples = min(self.all_subject_brain_data.shape[2] * self.all_subject_brain_data.shape[0], all_audio_data.shape[0])

        # tranpose channel and num_samples dims
        self.all_subject_brain_data = torch.transpose(self.all_subject_brain_data, 1, 2).reshape((-1, 61, T_out))[:self.num_samples, :, :]
        self.all_audio_data = all_audio_data[:self.num_samples]

        # for subject specific layer
        self.subject_num = torch.arange(num_subjects).repeat_interleave(self.num_samples//num_subjects)

        print("Brain data shape: " + str(self.all_subject_brain_data.shape))
        print("Waveform shape: " + str(all_audio_data.shape))
        print("Subject num shape: " + str(self.subject_num.shape))
        print("Audio sampling rate (Hz): " + str(self.audio_sample_rate))
        print("Brain data sampling rate (Hz): 500")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.all_subject_brain_data[idx], self.all_audio_data[idx], self.subject_num[idx])

    def get_audio_sample_rate(self):
        return self.audio_sample_rate

    def get_brain_data_sample_rate(self):
        return 500