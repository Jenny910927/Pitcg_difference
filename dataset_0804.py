import torch
from torch.utils.data import Dataset

import librosa
import os
import numpy as np
import random
import time
import json
import pickle
import argparse

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


def get_feature(audio_path):
    # TODO: Load audio from audio_path and generate feature.
    # return a 2-d (or maybe 3-d) array with shape (n, d) or (c, n, d), where n is the total frame number.
    y, sr = librosa.core.load(audio_path, mono=True)
    y_voc, y_acc = do_svs_spleeter(y, sr)

    y_acc = librosa.core.resample(y=y_acc, orig_sr=44100, target_sr=25600)

    feature = np.abs(librosa.cqt(y_acc, sr=25600, hop_length=256, fmin=librosa.midi_to_hz(24)
        , n_bins=96*4, bins_per_octave=12*4, filter_scale=1)).T
    # print (feature.shape)
    return feature
    # print (y.shape)

def process_pitch_and_note(json_path, feature_length):

    with open(json_path) as json_data:
        gt = json.load(json_data)
    pitch = gt["pitch"] #(23446, 2)
    notes = gt["st"] #(350, 2)
    
    # TODO: Match pitch and notes, return three lists: score_pitch (the pitch of the note), pitch_diff, "is_inlier"
    # inlier[i] = True if the frame i contains note, and the note prediction of frame i is correct.
    # The frame size of this function should be the same as the frame size of get_feature function.
    
    # print (feature_length)
    score_pitch = []
    is_inlier = []
    pitch_diff = []
    former_note = []
    next_note = []
    former_distance = []
    latter_distance = []

    cur_offset = 0
    # for j in range(1):
    for j in range(len(notes)):
        a = int(round(notes[j][0]*100.0))
        b = int(round(notes[j][1]*100.0))
        k = []

        for i in range(cur_offset, a):
            is_inlier.append(False)
            pitch_diff.append(0)
            score_pitch.append(0)
            former_note.append(0)
            next_note.append(0)
            former_distance.append(0)
            latter_distance.append(0)

        cur_offset = b

        # k = [pitch[a][1],]

        # bool_ = 0
        
        pitch_diff_abs = []

        for i in range(a, b):
            if pitch[i][1] > 0:
                k.append(pitch[i][1])
                pitch_diff_abs.append(abs(notes[j][2]-pitch[i][1]))
        k = np.array(k)

        # print (k)
        pitch_med = np.median(k)
        pitch_max = np.max(k)
        pitch_min = np.min(k)

        if (pitch_min <= notes[j][2] or pitch_max >= notes[j][2]) and max(pitch_diff_abs) <= 3:
            for i in range(a, b):
                if pitch[i][1] > 0:
                    is_inlier.append(True)
                    pitch_diff.append(notes[j][2]-pitch[i][1])
                    score_pitch.append(notes[j][2])
                    former_note.append(notes[j-1][2])
                    if j != len(notes)-1:
                        next_note.append(notes[j+1][2])
                    else:
                        next_note.append(0)
                    former_distance.append(i-a)
                    latter_distance.append(b-i)

                else:
                    is_inlier.append(False)
                    pitch_diff.append(0)
                    score_pitch.append(0)
                    former_note.append(0)
                    next_note.append(0)
                    former_distance.append(0)
                    latter_distance.append(0)
        else:
            for i in range(a, b):
                is_inlier.append(False)
                pitch_diff.append(0)
                score_pitch.append(0)
                former_note.append(0)
                next_note.append(0)
                former_distance.append(0)
                latter_distance.append(0)

        # print (pitch_med, pitch_max, pitch_min) 
        # print (notes[j])
        
    
        # score_pitch.append(notes[j][2])
        # pitch_diff.append(abs(score_pitch-pitch_med))
    # print (cur_offset)
    for i in range(cur_offset, feature_length):
        is_inlier.append(False)
        pitch_diff.append(0)
        score_pitch.append(0)
        former_note.append(0)
        next_note.append(0)
        former_distance.append(0)
        latter_distance.append(0)

    
    score_pitch = np.array(score_pitch)
    pitch_diff = np.array(pitch_diff)
    is_inlier = np.array(is_inlier)
    former_note = np.array(former_note)
    next_note = np.array(next_note)
    former_distance = np.array(former_distance)
    latter_distance = np.array(latter_distance)

    print (score_pitch.shape)
    print (pitch_diff.shape)
    print (is_inlier.shape)
    print (former_note.shape)
    print (next_note.shape)
    print (former_distance.shape)
    print (latter_distance.shape)

    # (23446, n)
    return (score_pitch, pitch_diff, is_inlier, former_note, next_note, former_distance, latter_distance)
    
    
class PitchDiffDataset(Dataset):
    def __init__(self, json_paths, audio_paths):
        self.features = []
        self.score_pitch = []
        self.pitch_diff = []
        self.is_inlier = []
        self.former_note = []
        self.next_note = []
        self.former_distance = []
        self.latter_distance = []

        for i in range(len(json_paths)):

            json_path = json_paths[i]
            audio_path = audio_paths[i]

            features = get_feature(audio_path)
            features = np.array(features)
            score_pitch, pitch_diff, is_inlier, former_note, next_note, former_distance, latter_distance = process_pitch_and_note(json_path, features.shape[-2])

            # print (score_pitch)
            # print (pitch_diff)

            self.features.append(features) #(23446, 384)
            self.score_pitch.append(score_pitch) #(23446,)
            self.pitch_diff.append(pitch_diff) #(23446,)
            self.is_inlier.append(is_inlier) #(23446,)
            self.former_note.append(former_note)
            self.next_note.append(next_note)
            self.former_distance.append(former_distance)
            self.latter_distance.append(latter_distance)

    def __getitem__(self, idx):
        return (self.features[idx], self.score_pitch[idx], self.pitch_diff[idx], self.is_inlier[idx], self.former_note[idx], self.next_note[idx], self.former_distance[idx], self.latter_distance[idx])

    def __len__(self):
        return len(self.features)
