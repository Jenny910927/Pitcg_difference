import sys
import os
import time
import argparse
import torch


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
        # print ("use", device)
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        predictor = predictor.EffNetPredictor(device=device, model_path=args.model_path)
        # predictor = predictor.EffNetPredictor(model_path=model_path)
        song_id = '1'
        results = {}
        do_svs = args.svs

        predict_sequential.predict_one_song(predictor, wav_path, song_id, results, do_svs=do_svs
            , tomidi=True, output_path=output_path, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres), beat_refine=beat_refine)


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
