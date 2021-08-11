from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
from .audio_dataset import get_feature, get_all_feature
# import torchcrepe

def do_svs_spleeter(y, sr):
    from spleeter.separator import Separator
    import warnings
    separator = Separator('spleeter:2stems')
    warnings.filterwarnings('ignore')

    if sr != 44100:
        y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)

    waveform = np.expand_dims(y, axis= 1)

    prediction = separator.separate(waveform)
    # print (prediction["vocals"].shape)
    ret_voc = librosa.core.to_mono(prediction["vocals"].T)
    ret_voc = np.clip(ret_voc, -1.0, 1.0)

    ret_acc = librosa.core.to_mono(prediction["accompaniment"].T)
    ret_acc = np.clip(ret_acc, -1.0, 1.0)
    del separator

    return ret_voc, ret_acc


class SeqDataset(Dataset):
    def __init__(self, wav_path, song_id, do_svs= False):

        # if mix_path == None:
        y, sr = librosa.core.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
        y = librosa.util.normalize(y)
        # else:
        #     y, sr = librosa.core.load(mix_path, sr=None, mono=True)
        #     svs_y, sr2 = librosa.core.load(wav_path, sr=None, mono=True)

        # if True:
        # if do_svs == True:
        y_voc, y_acc = do_svs_spleeter(y, 44100)
            # import soundfile
            # soundfile.write("vocal.wav", y, 44100, subtype='PCM_16')
            # y, sr = librosa.core.load("vocal.wav", sr=None, mono=True)

        self.data_instances = []
        self.vocal = y_voc

        cqt_data = get_all_feature(y_voc, y_acc, isfile=False)

        print (cqt_data.shape)

        frame_size = 1024.0 / 44100.0

        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
        # print (cqt_size, frame_num)

        width = frame_num
        my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

        # for frame_idx in range(frame_num):
        for frame_idx in range(0, frame_num, width):

            cqt_feature = []
            for frame_window_idx in range(frame_idx, frame_idx + width):
            # for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                # Boundary check
                # if frame_window_idx < 0:
                #     choosed_idx = 0
                # elif frame_window_idx >= frame_num:
                #     choosed_idx = frame_num - 1
                # else:
                #     choosed_idx = frame_window_idx

                if frame_window_idx < 0 or frame_window_idx >= frame_num:
                    cqt_feature.append(my_padding.unsqueeze(1))
                else:
                    choosed_idx = frame_window_idx
                    cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

                # cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

            cqt_feature = torch.cat(cqt_feature, dim=1)

            self.data_instances.append((cqt_feature, song_id))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
