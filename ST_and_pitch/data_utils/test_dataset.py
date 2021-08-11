from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
from .audio_dataset import *

class TestDataset(Dataset):
    """
    A TestDataset contains ALL frames for all songs, which is different from AudioDataset.
    """

    def __init__(self, data_dir, mix_trail=None, mix_only=True):
        self.data_instances = []

        for song_dir in tqdm(sorted(Path(data_dir).iterdir())):

            if mix_only == False:
                wav_path = os.path.join(song_dir, "Vocal.wav")
                # gt_path = os.path.join(song_dir, "gt_1005.txt")
                pitch_path = os.path.join(song_dir, "pitch.txt")
                orig_wav_path = os.path.join(song_dir, "Inst.wav")
                
                if not os.path.isfile(orig_wav_path):
                    orig_wav_path = None

                cqt_data = get_all_feature(wav_path, orig_wav_path, test=True)

            else:
                mix_path = os.path.join(song_dir, mix_trail)
                y, sr = librosa.core.load(mix_path, sr=None, mono=True)
                if sr != 44100:
                    y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)
                y = librosa.util.normalize(y)

                cqt_data = get_feature(y, sr=44100)

            song_id = song_dir.stem

            # Load song and extract features
            
            # y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            # cqt_data, mfcc_data = get_feature(y, sr)
            # cqt_data = get_feature(y, sr)


            # pitch_data = np.loadtxt(pitch_path)
            frame_size = 1024.0 / 44100.0
            # pitch_data = do_interpolation(pitch_data, frame_size, cqt_data.shape[0])

            # For each frame, combine adjacent frames as a data_instance

            # frame_num, cqt_size, mfcc_size = cqt_data.shape[0], cqt_data.shape[1], mfcc_data.shape[1]
            # frame_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1]
            frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
            my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)
            # print (cqt_size, frame_num)
            width = 6400
            # for frame_idx in range(frame_num):
            for frame_idx in range(0, frame_num, width):
                cqt_feature = []
                pitch_feature = []
                for frame_window_idx in range(frame_idx, frame_idx + width):
                # for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                    # Boundary check

                    if frame_window_idx < 0 or frame_window_idx >= frame_num:
                        cqt_feature.append(my_padding.unsqueeze(1))
                        pitch_feature.append(1.0)
                    else:
                        choosed_idx = frame_window_idx
                        cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
                        pitch_feature.append(1.0)

                    # if frame_window_idx < 0:
                    #     choosed_idx = 0
                    # elif frame_window_idx >= frame_num:
                    #     choosed_idx = frame_num - 1
                    # else:
                    #     choosed_idx = frame_window_idx

                    # cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
                    # pitch_feature.append(1.0)

                cqt_feature = torch.cat(cqt_feature, dim=1)

                # mfcc_feature = []
                # for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                #     # Boundary check
                #     if frame_window_idx < 0:
                #         choosed_idx = 0
                #     elif frame_window_idx >= frame_num:
                #         choosed_idx = frame_num - 1
                #     else:
                #         choosed_idx = frame_window_idx

                #     mfcc_feature.append(mfcc_data[choosed_idx].unsqueeze(1))

                # mfcc_feature = torch.cat(mfcc_feature, dim=1)

                # print (cqt_feature.shape)
                # print (mfcc_feature.shape)
                
                # self.data_instances.append((cqt_feature, song_id))
                self.data_instances.append((cqt_feature, torch.tensor(pitch_feature, dtype=torch.float), song_id))

        print('Dataset initialized from {}.'.format(data_dir))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
