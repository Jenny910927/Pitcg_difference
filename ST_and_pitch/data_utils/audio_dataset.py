from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import librosa
import os
import numpy as np
import random
import concurrent.futures
import h5py
import time
import json
import pickle

def get_cqt(y, filter_scale=1):

    # cqt = librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(24)
    #     , n_bins=96*4, bins_per_octave=12*4, filter_scale=filter_scale).T
    # return cqt.real, cqt.imag
    return np.abs(librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(24)
        , n_bins=96*4, bins_per_octave=12*4, filter_scale=filter_scale)).T

def get_mfcc(y):
    mfcc = librosa.feature.mfcc(y, sr=44100, n_fft=1024, n_mfcc=20, hop_length=1024, center=True, pad_mode='reflect')
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0).T
    return mfcc_feature

def get_feature(y, sr):
    # y = librosa.util.normalize(y)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future1_1 = executor.submit(get_cqt, y, 0.5)
        future1_2 = executor.submit(get_cqt, y, 1.0)
        future1_3 = executor.submit(get_cqt, y, 2.0)

        cqt_feature1_1 = future1_1.result()
        cqt_feature1_2 = future1_2.result()
        cqt_feature1_3 = future1_3.result()

        # cqt_feature1_1, cqt_feature1_1_2 = future1_1.result()
        # cqt_feature1_2, cqt_feature1_2_2 = future1_2.result()
        # cqt_feature1_3, cqt_feature1_3_2 = future1_3.result()

        cqt_feature1_1 = torch.tensor(cqt_feature1_1, dtype=torch.float).unsqueeze(1)
        cqt_feature1_2 = torch.tensor(cqt_feature1_2, dtype=torch.float).unsqueeze(1)
        cqt_feature1_3 = torch.tensor(cqt_feature1_3, dtype=torch.float).unsqueeze(1)
        cqt_feature1 = torch.cat((cqt_feature1_1, cqt_feature1_2, cqt_feature1_3), dim=1)

        # cqt_feature1_1_2 = torch.tensor(cqt_feature1_1_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_2_2 = torch.tensor(cqt_feature1_2_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_3_2 = torch.tensor(cqt_feature1_3_2, dtype=torch.float).unsqueeze(1)

        # cqt_feature1 = torch.cat((cqt_feature1, cqt_feature1_1_2, cqt_feature1_2_2, cqt_feature1_3_2), dim=1)

    return cqt_feature1
    # cqt_feature2 = get_cqt(y, 1.0)
    # cqt_feature = torch.tensor(cqt_feature, dtype=torch.float).unsqueeze(1)
    # cqt_feature2 = torch.tensor(cqt_feature2, dtype=torch.float).unsqueeze(1)
    # cqt_feature3 = torch.tensor(cqt_feature3, dtype=torch.float).unsqueeze(1)


    # cqt_feature = torch.cat((cqt_feature, cqt_feature2, cqt_feature3), dim=1)
    # mfcc_feature = torch.tensor(mfcc_feature, dtype=torch.float)

    # feature = np.abs(librosa.core.stft(y, n_fft=2048, hop_length=441*3, center=True))[0:256]
    # feature = librosa.feature.melspectrogram(y, n_fft=2048, hop_length=441*3, center=True, n_mels=512, fmax=8000)
    # return cqt_feature2

def get_all_feature(wav_path, acc_path, test=False, isfile=True):

    if isfile == True:
        y, sr = librosa.core.load(wav_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y= y, orig_sr= sr, target_sr= 44100)

    else:
        # wav_path is vocal part, while acc_path is acc part.
        y = wav_path
    
    if isfile == True:
        y2, sr2 = librosa.core.load(acc_path, sr=None, mono=True)
        if sr2 != 44100:
            y2 = librosa.core.resample(y= y2, orig_sr= sr2, target_sr= 44100)
    else:
        # wav_path is vocal part, while acc_path is acc part.
        y2 = acc_path
        

    max_mag = np.max(np.abs(np.add(y, y2)))
    y = y / (max_mag+0.0001)
    y2 = y2 / (max_mag+0.0001)

    # scale_f = (random.random() * 0.9 + 0.1)
    # if test == False:
    #     y = y * scale_f
    #     y2 = y2 * scale_f

    y_voc = y
    y_mix = np.add(y, y2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # future = executor.submit(get_feature, y, sr)

        # if orig_wav_path != None:
        #     future2 = executor.submit(get_feature, np.add(y, y2), sr)
        # else:
        #     future2 = executor.submit(get_feature, y, sr)
        # cqt_data = future.result()
        # cqt_data2 = future2.result()



        future2_1 = executor.submit(get_cqt, y_voc, 0.5)
        future2_2 = executor.submit(get_cqt, y_voc, 1.0)
        future2_3 = executor.submit(get_cqt, y_voc, 2.0)
        future1_1 = executor.submit(get_cqt, y_mix, 0.5)
        future1_2 = executor.submit(get_cqt, y_mix, 1.0)
        future1_3 = executor.submit(get_cqt, y_mix, 2.0)
        # future2 = executor.submit(get_mfcc, y)

        # cqt_feature, _ = future.result()

        cqt_feature2_1 = future2_1.result()
        # print (cqt_feature2_1.shape)
        cqt_feature2_2 = future2_2.result()
        cqt_feature2_3 = future2_3.result()
        cqt_feature1_1 = future1_1.result()
        cqt_feature1_2 = future1_2.result()
        cqt_feature1_3 = future1_3.result()

        # cqt_feature2_1, cqt_feature2_1_2 = future2_1.result()
        # cqt_feature2_2, cqt_feature2_2_2 = future2_2.result()
        # cqt_feature2_3, cqt_feature2_3_2 = future2_3.result()
        # cqt_feature1_1, cqt_feature1_1_2 = future1_1.result()
        # cqt_feature1_2, cqt_feature1_2_2 = future1_2.result()
        # cqt_feature1_3, cqt_feature1_3_2 = future1_3.result()

        cqt_feature2_1 = torch.tensor(cqt_feature2_1, dtype=torch.float).unsqueeze(1)
        cqt_feature2_2 = torch.tensor(cqt_feature2_2, dtype=torch.float).unsqueeze(1)
        cqt_feature2_3 = torch.tensor(cqt_feature2_3, dtype=torch.float).unsqueeze(1)

        cqt_feature2 = torch.cat((cqt_feature2_1, cqt_feature2_2, cqt_feature2_3), dim=1)

        # cqt_feature2_1_2 = torch.tensor(cqt_feature2_1_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature2_2_2 = torch.tensor(cqt_feature2_2_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature2_3_2 = torch.tensor(cqt_feature2_3_2, dtype=torch.float).unsqueeze(1)

        # cqt_feature2 = torch.cat((cqt_feature2, cqt_feature2_1_2, cqt_feature2_2_2, cqt_feature2_3_2), dim=1)


        cqt_feature1_1 = torch.tensor(cqt_feature1_1, dtype=torch.float).unsqueeze(1)
        cqt_feature1_2 = torch.tensor(cqt_feature1_2, dtype=torch.float).unsqueeze(1)
        cqt_feature1_3 = torch.tensor(cqt_feature1_3, dtype=torch.float).unsqueeze(1)

        cqt_feature1 = torch.cat((cqt_feature1_1, cqt_feature1_2, cqt_feature1_3), dim=1)

        # cqt_feature1_1_2 = torch.tensor(cqt_feature1_1_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_2_2 = torch.tensor(cqt_feature1_2_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_3_2 = torch.tensor(cqt_feature1_3_2, dtype=torch.float).unsqueeze(1)

        # cqt_feature1 = torch.cat((cqt_feature1, cqt_feature1_1_2, cqt_feature1_2_2, cqt_feature1_3_2), dim=1)

        # cqt_feature2_1 = torch.tensor(cqt_feature2_1, dtype=torch.float).unsqueeze(1)
        # cqt_feature2_2 = torch.tensor(cqt_feature2_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature2_3 = torch.tensor(cqt_feature2_3, dtype=torch.float).unsqueeze(1)
        # cqt_feature2 = torch.cat((cqt_feature2_1, cqt_feature2_2, cqt_feature2_3), dim=1)

        # cqt_feature1_1 = torch.tensor(cqt_feature1_1, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_2 = torch.tensor(cqt_feature1_2, dtype=torch.float).unsqueeze(1)
        # cqt_feature1_3 = torch.tensor(cqt_feature1_3, dtype=torch.float).unsqueeze(1)
        # cqt_feature1 = torch.cat((cqt_feature1_1, cqt_feature1_2, cqt_feature1_3), dim=1)

    cqt_data = torch.cat((cqt_feature1, cqt_feature2), dim=1)
    # cqt_data = torch.cat((cqt_data, cqt_data2), dim=1)
    return cqt_data


class AudioDataset(Dataset):

    def __init__(self, gt_path=None, data_dir=None, h5py_path=None, output_path=None, label_path=None):
        self.data_instances = None
        self.instance_key = None
        self.instance_length = None

        if h5py_path is not None:
            self.h5py_path = h5py_path
            with h5py.File(self.h5py_path, 'r') as file:
                self.data_length = len(file["data"])

            with open(label_path, 'rb') as f:
                self.on_off_label, self.answer_label, self.gt_length, self.on_off_answer = pickle.load(f)

        else:
            with open(gt_path) as json_data:
                gt = json.load(json_data)

            self.data_instances = []
            self.instance_key = []
            self.instance_length = []

            total_count = len(os.listdir(data_dir))
            self.temp_cqt = {}
            future = {}
            print (time.time())
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for the_dir in os.listdir(data_dir):
                    wav_path = os.path.join(data_dir, the_dir, "Vocal.wav")
                    orig_wav_path = os.path.join(data_dir, the_dir, "Inst.wav")
                    future[the_dir] = executor.submit(get_all_feature, wav_path, orig_wav_path)
                    # future[the_dir] = executor.submit(get_all_feature_s3prl, wav_path, orig_wav_path)

                for the_dir in os.listdir(data_dir):
                    self.temp_cqt[the_dir] = future[the_dir].result()
                    # print (self.temp_cqt[the_dir].shape)
            print (time.time())

            for the_dir in tqdm(os.listdir(data_dir)):
                # gt_path = os.path.join(data_dir, the_dir, "gt.txt")
                # pitch_path = os.path.join(data_dir, the_dir, "pitch.txt")

                # cqt_data = get_all_feature(wav_path, orig_wav_path)
                cqt_data = self.temp_cqt[the_dir].permute(1, 0, 2)

                # gt_data = np.loadtxt(gt_path)
                gt_data = np.array(gt[the_dir])

                # pitch_data = np.loadtxt(pitch_path)
                frame_size = 1024.0 / 44100.0
                # frame_size = 0.01
                # print (gt_data)
                # print (answer_data)

                # pitch_data = do_interpolation_and_purify(pitch_data, answer_data, frame_size, cqt_data.shape[0])
                
                # frame_num, cqt_size, mfcc_size = cqt_data.shape[0], cqt_data.shape[1], mfcc_data.shape[1]
                # print (cqt_data.shape)
                frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
                
                # frame_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1]

                # print (cqt_size, frame_num)
                # width = 1600
                # # for frame_idx in range(frame_num):
                # my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

                # # this means silent frame
                # padding_gt = np.array([0, 1, 4, 12, 0])

                # for frame_idx in range(0, frame_num, width):
                #     cqt_feature = []
                #     pitch_feature = []
                #     answer_feature = []
                #     start = min(frame_idx, frame_num - width)
                #     for frame_window_idx in range(start, start + width):
                #     # for frame_window_idx in range(frame_idx, frame_idx + width):
                #     # for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                #         # Boundary check
                #         # if frame_window_idx < 0:
                #         #     choosed_idx = 0
                #         # elif frame_window_idx >= frame_num:
                #         #     choosed_idx = frame_num - 1
                #         # else:
                #         #     choosed_idx = frame_window_idx

                #         if frame_window_idx < 0 or frame_window_idx >= frame_num:
                #             cqt_feature.append(my_padding.unsqueeze(1))
                #             pitch_feature.append(0)
                #             answer_feature.append(padding_gt)
                #         else:
                #             choosed_idx = frame_window_idx
                #             cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
                #             pitch_feature.append(0)
                #             answer_feature.append(answer_data[choosed_idx])

                #         # cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

                #     cqt_feature = torch.cat(cqt_feature, dim=1)


                    # self.data_instances.append((cqt_feature, torch.tensor(pitch_data[frame_idx], dtype=torch.float)
                    #     , torch.tensor(answer_data[frame_idx], dtype=torch.long)))
                    # self.data_instances.append(cqt_feature)
                    # self.pitch_instances.append(torch.tensor(pitch_feature, dtype=torch.float))
                    # self.answer_instances.append(torch.tensor(answer_feature, dtype=torch.long))

                self.data_instances.append(cqt_data)
                # self.pitch_instances.append(1)
                # self.answer_instances.append(all_dist)
                # self.note_length.append(len(gt_data))
                # self.answer_instances_orig.append(answer_data)
                self.instance_key.append(the_dir)
                self.instance_length.append(cqt_data.shape[1])
            # augmented data
            """
            self.temp_cqt = {}
            future = {}
            print (time.time())
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                for the_dir in os.listdir(data_dir):
                    wav_path = os.path.join(data_dir, the_dir, "Vocal_m4.wav")
                    orig_wav_path = os.path.join(data_dir, the_dir, "Inst_m4.wav")
                    future[the_dir] = executor.submit(get_all_feature, wav_path, orig_wav_path)

                for the_dir in os.listdir(data_dir):
                    self.temp_cqt[the_dir] = future[the_dir].result()
            print (time.time())

            for the_dir in tqdm(os.listdir(data_dir)):
                # wav_path = os.path.join(data_dir, the_dir, "Vocal_m4.wav")
                # gt_path = os.path.join(data_dir, the_dir, "gt_1005.txt")
                # pitch_path = os.path.join(data_dir, the_dir, "pitch.txt")
                # orig_wav_path = os.path.join(data_dir, the_dir, "Inst_m4.wav")

                # cqt_data = get_all_feature(wav_path, orig_wav_path)
                cqt_data = self.temp_cqt[the_dir]

                # gt_data = np.loadtxt(gt_path)
                gt_data = gt[the_dir]
                answer_data = preprocess(gt_data, cqt_data.shape[0], pitch_shift=-4)


                pitch_data = np.loadtxt(pitch_path)
                frame_size = 1024.0 / 44100.0
                pitch_data = do_interpolation_and_purify(pitch_data, answer_data, frame_size, cqt_data.shape[0])
                
                # frame_num, cqt_size, mfcc_size = cqt_data.shape[0], cqt_data.shape[1], mfcc_data.shape[1]
                # print (cqt_data.shape)
                frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
                
                # frame_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1]

                # print (cqt_size, frame_num)
                width = 1600
                # for frame_idx in range(frame_num):
                my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

                # this means silent frame
                padding_gt = np.array([0, 1, 4, 12, 0])

                for frame_idx in range(0, frame_num, width):
                    cqt_feature = []
                    pitch_feature = []
                    answer_feature = []
                    
                    start = min(frame_idx, frame_num - width)
                    for frame_window_idx in range(start, start + width):
                    # for frame_window_idx in range(frame_idx, frame_idx + width):
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
                            pitch_feature.append(0)
                            answer_feature.append(padding_gt)
                        else:
                            choosed_idx = frame_window_idx
                            cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
                            pitch_feature.append(pitch_data[choosed_idx] - 4.0)
                            answer_feature.append(answer_data[choosed_idx])

                        # cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

                    cqt_feature = torch.cat(cqt_feature, dim=1)


                    # self.data_instances.append((cqt_feature, torch.tensor(pitch_data[frame_idx], dtype=torch.float)
                    #     , torch.tensor(answer_data[frame_idx], dtype=torch.long)))
                    self.data_instances.append(cqt_feature)
                    self.pitch_instances.append(torch.tensor(pitch_feature, dtype=torch.float))
                    self.answer_instances.append(torch.tensor(answer_feature, dtype=torch.long))

            """

            
            print (self.data_instances[0].shape)
            print('Dataset initialized from {}.'.format(data_dir))


            with h5py.File(output_path, "w") as f:
                grp = f.create_group("data")
                # grp2 = f.create_group("pitch")
                grp2 = f.create_group("instance_length")

                for i in range(len(self.data_instances)):
                    grp.create_dataset(str(self.instance_key[i]), data=self.data_instances[i])
                    # grp2.create_dataset(str(i), data=self.pitch_instances[i])
                    grp2.create_dataset(str(self.instance_key[i]), data=self.instance_length[i])

                # dset = f.create_dataset("data", (len(self.data_instances), self.data_instances[0].shape[0], self.data_instances[0].shape[1]
                #     , self.data_instances[0].shape[2]))
                # dset2 = f.create_dataset("pitch", (len(self.pitch_instances), self.pitch_instances[0].shape[0]))
                # dset3 = f.create_dataset("answer", (len(self.answer_instances), self.answer_instances[0].shape[0], self.answer_instances[0].shape[1]))

                # for i in range(len(self.data_instances)):
                #     dset[i] = self.data_instances[i]
                #     dset2[i] = self.pitch_instances[i]
                #     dset3[i] = self.answer_instances[i]


    def __getitem__(self, idx):
        if self.data_instances is None:
            self.data_instances = h5py.File(self.h5py_path, 'r')["data"]
            # self.pitch_instances = h5py.File(self.h5py_path, 'r')["pitch"]
            self.instance_length = h5py.File(self.h5py_path, 'r')["instance_length"]
            self.instance_key = list(h5py.File(self.h5py_path, 'r')["data"].keys())

        cur_key = self.instance_key[idx]
        return (self.data_instances[cur_key][()], self.instance_length[cur_key][()]
            , self.on_off_label[cur_key], self.answer_label[cur_key]
            , self.gt_length[cur_key], self.on_off_answer[cur_key])
        # return (self.data_instances[str(idx)][()], self.pitch_instances[str(idx)][()], self.answer_instances[str(idx)][()])
        # return self.data_instances[idx]

    def __len__(self):
        # return len(self.data_instances)
        return self.data_length



# class AudioDataset(Dataset):

#     def __init__(self, data_dir, is_test=False):
#         self.data_instances = []
#         for the_dir in tqdm(os.listdir(data_dir)):
#             wav_path = os.path.join(data_dir, the_dir, "Vocal.wav")
#             gt_path = os.path.join(data_dir, the_dir, "gt_1005.txt")
#             pitch_path = os.path.join(data_dir, the_dir, "pitch.txt")
#             orig_wav_path = os.path.join(data_dir, the_dir, "Inst.wav")

#             cqt_data = get_all_feature(wav_path, orig_wav_path)

#             gt_data = np.loadtxt(gt_path)
#             answer_data = preprocess(gt_data, cqt_data.shape[0])


#             pitch_data = np.loadtxt(pitch_path)
#             frame_size = 1024.0 / 44100.0
#             pitch_data = do_interpolation_and_purify(pitch_data, answer_data, frame_size, cqt_data.shape[0])
            
#             # frame_num, cqt_size, mfcc_size = cqt_data.shape[0], cqt_data.shape[1], mfcc_data.shape[1]
#             # print (cqt_data.shape)
#             frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
            
#             # frame_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1]

#             # print (cqt_size, frame_num)
#             width = 6400
#             # for frame_idx in range(frame_num):
#             my_padding = torch.zeros((cqt_data.shape[1], cqt_data.shape[2]), dtype=torch.float)

#             # this means silent frame
#             padding_gt = np.array([0, 1, 4, 12, 0])

#             for frame_idx in range(0, frame_num, width):
#                 cqt_feature = []
#                 pitch_feature = []
#                 answer_feature = []
#                 for frame_window_idx in range(frame_idx, frame_idx + width):
#                 # for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
#                     # Boundary check
#                     # if frame_window_idx < 0:
#                     #     choosed_idx = 0
#                     # elif frame_window_idx >= frame_num:
#                     #     choosed_idx = frame_num - 1
#                     # else:
#                     #     choosed_idx = frame_window_idx

#                     if frame_window_idx < 0 or frame_window_idx >= frame_num:
#                         cqt_feature.append(my_padding.unsqueeze(1))
#                         pitch_feature.append(0)
#                         answer_feature.append(padding_gt)
#                     else:
#                         choosed_idx = frame_window_idx
#                         cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))
#                         pitch_feature.append(pitch_data[choosed_idx])
#                         answer_feature.append(answer_data[choosed_idx])

#                     # cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

#                 cqt_feature = torch.cat(cqt_feature, dim=1)


#                 # self.data_instances.append((cqt_feature, torch.tensor(pitch_data[frame_idx], dtype=torch.float)
#                 #     , torch.tensor(answer_data[frame_idx], dtype=torch.long)))
#                 self.data_instances.append((cqt_feature, torch.tensor(pitch_feature, dtype=torch.float)
#                     , torch.tensor(answer_feature, dtype=torch.long)))

#         print('Dataset initialized from {}.'.format(data_dir))

#     def __getitem__(self, idx):
#         return self.data_instances[idx]

#     def __len__(self):
#         return len(self.data_instances)





# class AudioDataset(Dataset):

#     def __init__(self, data_dir, is_test=False):
#         self.data_instances = []
#         for the_dir in tqdm(os.listdir(data_dir)):
#             wav_path = data_dir + "/" + the_dir + "/Vocal.wav"
#             gt_path = data_dir + "/" + the_dir + "/" + "gt.txt"

#             y, sr = librosa.core.load(wav_path, sr=None, mono=True)
            
#             data = get_feature(y, sr)

#             gt_data = np.loadtxt(gt_path)
#             answer_data = preprocess(gt_data, data[0].shape[0])
#             # np.set_printoptions(threshold=50000)
#             # if str(the_dir) == '1':
#             #     print (answer_data)
#             # print (data.shape)
#             # print(answer_data.shape)
#             # print (answer_data[1000])
#             self.data_instances.append((data, torch.tensor(answer_data, dtype=torch.long)))

#         print('Dataset initialized from {}.'.format(data_dir))

#     def __getitem__(self, idx):
#         # print (self.data_instances[idx][0].shape)
#         rand_num = int(random.random() * (len(self.data_instances[idx][0][0]) - 1))
#         # get (rand_num-3 to rand_num+ 3) frames
#         cqt_feature = torch.empty(11, len(self.data_instances[idx][0][0][0]), dtype=torch.float)

#         for i in range(rand_num - 5, rand_num + 6):
#             if i < 0:
#                 num = 0
#             elif i > len(self.data_instances[idx][0][0]) - 1:
#                 num = len(self.data_instances[idx][0][0]) - 1
#             else:
#                 num = i
#             cqt_feature[i - rand_num + 5, :] = self.data_instances[idx][0][0][num]


#         mfcc_feature = torch.empty(11, len(self.data_instances[idx][0][1]), dtype=torch.float)

#         # print (self.data_instances[idx][0][1].shape)

#         for i in range(rand_num - 5, rand_num + 6):
#             if i < 0:
#                 num = 0
#             elif i > len(self.data_instances[idx][0][0]) - 1:
#                 num = len(self.data_instances[idx][0][0]) - 1
#             else:
#                 num = i
#             mfcc_feature[i - rand_num + 5, :] = self.data_instances[idx][0][1][:, num]



#         return (cqt_feature, mfcc_feature, self.data_instances[idx][1][rand_num])

#         # use whole song as training data
#         # whole_seq_feature = torch.empty(len(self.data_instances[idx][0]), 7, len(self.data_instances[idx][0][0]), dtype=torch.float)
#         # whole_label = torch.empty(len(self.data_instances[idx][0]), len(self.data_instances[idx][1][0]), dtype=torch.long)

#         # for j in range(len(self.data_instances[idx][0])):
#         #     feature = torch.empty(7, len(self.data_instances[idx][0][0]), dtype=torch.float)

#         #     for i in range(j - 3, j + 4):
#         #         if i < 0:
#         #             num = 0
#         #         elif i > len(self.data_instances[idx][0]) - 1:
#         #             num = len(self.data_instances[idx][0]) - 1
#         #         else:
#         #             num = i
#         #         feature[i - j + 3, :] = torch.tensor(self.data_instances[idx][0][num])

#         #     whole_seq_feature[j, :, :] = feature.clone()
#         #     whole_label[j, :] = torch.tensor(self.data_instances[idx][1][j], dtype=torch.long)

#         # return (whole_seq_feature, whole_label)

#     def __len__(self):
#         return len(self.data_instances)