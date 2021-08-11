import argparse
import pickle
from predictor_2 import PitchDiffPredictor
import torch
from dataset_former_note import PitchDiffDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset')
    parser.add_argument('val_dataset')

    args = parser.parse_args()

    with open(args.train_dataset, 'rb') as f:
        train_pkl = pickle.load(f)

    with open(args.val_dataset, 'rb') as f:
        val_pkl = pickle.load(f)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    predictor = PitchDiffPredictor(train_pkl, val_pkl, device=device, pretrained_path=None)
    pitch_curve_pred, pitch_curve_gt, segments = predictor.fit(model_dir="model", save_prefix="pitch", plot_path="loss.png")
    predictor.plot_pitch_line(pitch_curve_gt, pitch_curve_pred, segments, plot_path="pitch.png")
    # predictor.plot_pitch_line(test_dataset=val_pkl, plot_path="pitch.png")