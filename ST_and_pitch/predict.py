import argparse
import json
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # nopep8
from data_utils import TestDataset
from predictor import EffNetPredictor
from pathlib import Path
import pickle
import time

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

def main(args):
    # Create predictor
    # print (time.time())
    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    # print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    predictor = EffNetPredictor(device=device, model_path=args.model_path)
    # print('Creating testing dataset...')

    filename_trail = '_0314_voc_and_mix.pkl'
    target_path = Path("./") / (Path(args.test_dir).stem + filename_trail)

    print (time.time())
    
    if os.path.isfile(args.test_dir):
        # read from pickle
        # test_dataset = joblib.load(target_path)
        with open(args.test_dir, 'rb') as f:
            test_dataset = pickle.load(f)

        print('Dataset loaded at {}.'.format(args.test_dir))

    else:
        # Read from test_dir
        test_dataset = TestDataset(args.test_dir, mix_trail="Mixture.m4a", mix_only=False)

        # joblib.dump(test_dataset, target_path, compress=3)
        with open(target_path, 'wb') as f:
            pickle.dump(test_dataset, f, protocol=4)

        print('Dataset generated at {}.'.format(target_path))

    # Feed dataset to the model
    print (time.time())
    # print('Predicting {}...'.format(args.test_dir))
    results = predictor.predict(test_dataset, show_tqdm= True, onset_thres=float(args.onset_thres), offset_thres=float(args.offset_thres))

    # Write results to target file
    with open(args.predict_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    print (time.time())
    # print('Prediction done. File writed to: {}'.format(args.predict_file))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('predict_file')
    parser.add_argument('model_path')
    parser.add_argument('-on', "--onset_thres", default=0.1)
    parser.add_argument('-off', "--offset_thres", default=0.5)
    parser.add_argument('-d', "--device", default="cuda:0")
    parser.add_argument('-q', '--quantize', action='store_true')

    args = parser.parse_args()

    main(args)
