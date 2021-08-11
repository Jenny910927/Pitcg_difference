import argparse
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from predictor import EffNetPredictor

from data_utils.seq_dataset import SeqDataset
from pathlib import Path
from tqdm import tqdm
import mido
import warnings
import numpy as np

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
# import tensorflow as tf
# from tempocnn.classifier import TempoClassifier
# from tempocnn.feature import read_features


def notes2mid_norefine(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0


    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        # note_on_length = int(round(note_on_length / (float(480/4)))) * (480/4)
        # note_off_length = int(round(note_off_length / (float(480/4)))) * (480/4)
        # note_on_length = int(note_on_length)
        # note_off_length = int(note_off_length)

        # print (note_on_length+1, note_off_length-1)

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
    

def convert_to_midi(predicted_result, song_id, output_path, shift, tempo):
    to_convert = predicted_result[song_id]
    # print (to_convert)
    # print (len(to_convert))
    if shift == 0.0 and tempo == 120.0:
        mid = notes2mid_norefine(to_convert)
    else:
        mid = notes2mid_refine(to_convert, shift, tempo)
    mid.save(output_path)

def predict_one_song(predictor, wav_path, song_id, results, do_svs, tomidi, output_path, onset_thres, offset_thres, show_tqdm=True, beat_refine=False):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test_dataset = SeqDataset(wav_path, song_id, do_svs=do_svs)

    results = predictor.predict(test_dataset, results=results, show_tqdm=show_tqdm, onset_thres=onset_thres, offset_thres=offset_thres)
    shift = 0.0
    tempo = 120.0
    if beat_refine == True:
        print ("beat refine!")
        results, shift, tempo = tempo_refine(results, wav_path)
    if tomidi == True:
        convert_to_midi(results, song_id, output_path, shift, tempo)

    return results, test_dataset.vocal

def main(args):
    # Create predictor
    warnings.filterwarnings("ignore")
    predictor = EffNetPredictor(model_path=args.model_path)

    results = {}
    print('Predicting {}...'.format(args.test_dir))
    count = 0
    for song_dir in tqdm(sorted(Path(args.test_dir).iterdir())):
        wav_path = str(song_dir / 'Vocal.wav')
        song_id = song_dir.stem

        if not os.path.isfile(wav_path):
            continue

        output_path = str(song_dir / 'trans.mid')

        results = predict_one_song(predictor, wav_path, song_id, results, do_svs=True
            , tomidi=False, output_path=None, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres), show_tqdm=False)
        count = count + 1
        # if count >= 100:
        #     break
    print (count)
        
    # Write results to target file
    with open(args.predict_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    print('Prediction done. File writed to: {}'.format(args.predict_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')
    parser.add_argument('-m', "--tomidi", action="store_true")
    parser.add_argument('-on', "--onset_thres", default=0.1)
    parser.add_argument('-off', "--offset_thres", default=0.5)

    args = parser.parse_args()

    main(args)
