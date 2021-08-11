import torch
import torch.nn as nn
import argparse
from predictor import EffNetPredictor
import os
from data_utils import AudioDataset

def main(args):

    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
        # torch.cuda.set_device(0)
    print ("use", device)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.backends.cudnn.deterministic = True
    
    predictor = EffNetPredictor(device=device, model_path=args.model_path)

    predictor.training_dataset = AudioDataset(h5py_path=args.training_dataset, label_path=args.training_label)
    predictor.validation_dataset = AudioDataset(h5py_path=args.validation_dataset, label_path=args.validation_label)

    predictor.fit(
        model_dir=args.model_dir,
        batch_size=1,
        valid_batch_size=1,
        epoch=100,
        lr=1e-4,
        save_every_epoch=1,
        save_prefix=args.save_prefix,
        plot_path=args.plot_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_dataset')
    parser.add_argument('training_label')
    parser.add_argument('validation_dataset')
    parser.add_argument('validation_label')
    parser.add_argument('model_dir')
    parser.add_argument('save_prefix')
    parser.add_argument('device')
    parser.add_argument('plot_path')
    parser.add_argument('--model-path')

    args = parser.parse_args()

    main(args)
