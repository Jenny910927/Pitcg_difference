import sys
import os
import time
import argparse
import torch
import numpy as np
import torchcrepe
import librosa
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def get_pitch(vocal, device):

    # y, sr = librosa.core.load(wav_path, sr=16000, mono=True)
    # if sr != 16000:
    audio = librosa.core.resample(y= vocal, orig_sr=44100, target_sr=16000)
    # sr = 16000

    audio = torch.tensor(np.copy(audio))[None]

    audio = audio.to(device)

    hop_length = 160

    fmin = 50 # C2 = 65.406 Hz
    fmax = 1000 # B5 = 987.77 Hz

    model = "full"

    pitch = torchcrepe.predict(audio, 16000, hop_length, fmin, fmax, model, batch_size=512, device=device).cpu().numpy()
    pitch_output = np.array([[i*0.01, librosa.hz_to_midi(pitch[0][i])] for i in range(pitch.shape[1])])

    return pitch_output


def main(args):
    model_path = args.model_path
    model_name = args.model_name
    wav_path = args.input
    output_path = args.output
    beat_refine = args.beat_refine

    if model_name == 'efficientnet':
        import predict_sequential
        import predictor

        device= 'cpu'
        if torch.cuda.is_available():
            device = args.device
        print ("use", device)
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        predictor = predictor.EffNetPredictor(device=device, model_path=args.model_path)
        # predictor = predictor.EffNetPredictor(model_path=model_path)
        song_id = '1'
        st_results = {}
        do_svs = args.svs

        st_results, vocal = predict_sequential.predict_one_song(predictor, wav_path, song_id, st_results, do_svs=do_svs
            , tomidi=False, output_path=output_path, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres), beat_refine=beat_refine)
        
        print ("computing pitch (using crepe~)", time.time())
        pitch_result = get_pitch(vocal, device)

        pitch_and_st = {}
        pitch_and_st["pitch"] = pitch_result.tolist()
        pitch_and_st["st"] = st_results[song_id]

        # Write results to target file
        with open(output_path, 'w') as f:
            output_string = json.dumps(pitch_and_st)
            f.write(output_string)

if __name__ == '__main__':
    print (time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-m', '--model_name', default='efficientnet')
    parser.add_argument('-mp', '--model_path', default='models/b0_e_20000')
    parser.add_argument('-s', '--svs', default=True)
    parser.add_argument('-b', '--beat_refine', default=False)
    parser.add_argument('-on', "--onset_thres", default=0.1)
    parser.add_argument('-off', "--offset_thres", default=0.5)
    parser.add_argument('-d', "--device", default="cuda:0")

    args = parser.parse_args()

    main(args)
    print (time.time())
