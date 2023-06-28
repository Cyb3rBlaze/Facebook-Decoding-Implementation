import os
import scipy.io
import torch
import torchaudio

from scipy.signal import butter, sosfilt
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    """Custom dataset used to load pairs for training.
    """
    def __init__(self, data_dir, T_out, num_subjects, exclude, val=False):

        all_subject_brain_data = None

        # for resampling purposes
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53

        self.audio_sample_rate = 16000
        self.brain_data_sample_rate = 50

        min_samples = float("inf")

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
                    final_filtered = torch.tensor(sosfilt(sos, channel))

                    final_filtered = torchaudio.functional.resample(final_filtered, 500, self.brain_data_sample_rate)
                    final_filtered = final_filtered[:final_filtered.shape[0]//(self.brain_data_sample_rate*3)*self.brain_data_sample_rate*3]

                    final_filtered = final_filtered.reshape((-1, 1, self.brain_data_sample_rate*3))

                    if brain_data == None:
                        brain_data = final_filtered
                    else:
                        brain_data = torch.cat((brain_data, final_filtered), dim=1)    # [num_samples, num_channels, time_steps]
                
                if i == 1:
                    all_subject_brain_data = torch.unsqueeze(brain_data, 0)
                else:
                    if min_samples > brain_data.shape[0]:
                        min_samples = brain_data.shape[0]
                    all_subject_brain_data = torch.vstack((all_subject_brain_data[:, :min_samples, :61, :self.brain_data_sample_rate*3], torch.unsqueeze(brain_data, 0)[:, :min_samples, :61, :self.brain_data_sample_rate*3]))
        
        initial_mean = torch.unsqueeze(torch.mean(all_subject_brain_data[:, :, :, :int(self.brain_data_sample_rate*0.5)], dim=3), 3)
        all_subject_brain_data = all_subject_brain_data - initial_mean

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

        all_audio_data = all_audio_data.view(-1, self.audio_sample_rate*3).repeat(num_subjects, 1, 1)

        self.num_samples = min(all_subject_brain_data.shape[1], all_audio_data.shape[1])

        if val == True:
            self.subject_num = torch.arange(num_subjects).repeat(self.num_samples, 1)[int(self.num_samples*0.8):self.num_samples].view(-1)
            self.all_subject_brain_data = all_subject_brain_data[:, int(self.num_samples*0.8):self.num_samples, :, :].reshape((-1, 61, self.brain_data_sample_rate*3))[:, :, :T_out]
            all_audio_data = all_audio_data[:, int(self.num_samples*0.8):self.num_samples, :]
        else:
            self.subject_num = torch.arange(num_subjects).repeat(self.num_samples, 1)[:int(self.num_samples*0.8)].view(-1)
            self.all_subject_brain_data = all_subject_brain_data[:, :int(self.num_samples*0.8), :, :].reshape((-1, 61, self.brain_data_sample_rate*3))[:, :, :T_out]
            all_audio_data = all_audio_data[:, :int(self.num_samples*0.8), :]

        self.all_audio_data = all_audio_data.reshape((-1, self.audio_sample_rate*3))

        for i, sample in tqdm(enumerate(self.all_subject_brain_data)):
            for j, channel_data in enumerate(sample):
                transformer = RobustScaler().fit(channel_data.numpy().reshape((-1, 1)))
                transformed_data = transformer.transform(channel_data.numpy().reshape((-1, 1)))
                self.all_subject_brain_data[i, j] = torch.tensor(transformed_data.reshape((-1)))
        
        self.all_subject_brain_data = torch.clip(self.all_subject_brain_data, -20, 20)

        print("Brain data shape: " + str(self.all_subject_brain_data.shape))
        print("Waveform shape: " + str(self.all_audio_data.shape))
        print("Subject num shape: " + str(self.subject_num.shape))
        print("Audio sampling rate (Hz): " + str(self.audio_sample_rate))
        print("Brain data sampling rate (Hz): " + str(self.brain_data_sample_rate))

    def __len__(self):
        return self.all_subject_brain_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.all_subject_brain_data[idx], self.all_audio_data[idx], self.subject_num[idx])

    def get_audio_sample_rate(self):
        return self.audio_sample_rate

    def get_brain_data_sample_rate(self):
        return self.brain_data_sample_rate