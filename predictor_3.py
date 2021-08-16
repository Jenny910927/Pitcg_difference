from net_RNN_0807 import MyNet
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np

class PitchDiffPredictor():
    def __init__(self, train_dataset, val_dataset, device, pretrained_path=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.model = MyNet().to(self.device)

        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            print('Model read from {}.'.format(pretrained_path))

    def draw_loss(self, training_loss_avg, validation_loss_avg, plot_path):
        training_loss_avg = np.array(training_loss_avg)
        validation_loss_avg = np.array(validation_loss_avg)
        epoch = np.arange(1, len(validation_loss_avg) + 1, 1)

        # print(validation_loss_avg.shape())


        plt.plot(epoch, training_loss_avg, color='b', label='Training_loss')
        plt.plot(epoch, validation_loss_avg, color='r', label='Validation_loss')

        plt.title("Loss", x=0.5, y=1.03)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend(loc = "best", fontsize=10)

        # plt.show()
        plt.savefig(plot_path, dpi=500)
        plt.close()



    def count_baseline(self):
        total_train_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            pitch_diff = batch[2][0].float()
            is_inlier = batch[3][0]

            pitch_diff_pred = torch.zeros(pitch_diff.shape)
            is_inlier = is_inlier.float().squeeze()
            loss = self.loss_fn(pitch_diff_pred, pitch_diff) * is_inlier
            loss_delta = []
            loss_delta_2 = []
            for i in range(loss.shape[0]-1):
                loss_delta.append(abs(loss[i]-loss[i+1]))
            loss_delta = np.array(loss_delta)
            for i in range(loss_delta.shape[0]-1):
                loss_delta_2.append(abs(loss_delta[i]-loss[i+1]))
            loss_delta_2 = np.array(loss_delta_2)

            loss = torch.sum(loss) / sum(is_inlier)
            loss_delta = np.sum(loss_delta)/sum(is_inlier)
            loss_delta_2 = np.sum(loss_delta_2)/sum(is_inlier)

            # print(loss.shape)
            total_train_loss = total_train_loss + loss.item() + loss_delta.item() + loss_delta_2.item()

        total_val_loss = 0
        for batch_idx, batch in enumerate(self.val_loader):
            pitch_diff = batch[2][0].float()
            is_inlier = batch[3][0]

            pitch_diff_pred = torch.zeros(pitch_diff.shape)
            is_inlier = is_inlier.float().squeeze()
            loss = self.loss_fn(pitch_diff_pred, pitch_diff) * is_inlier
            loss_delta = []
            loss_delta_2 = []
            for i in range(loss.shape[0]-1):
                loss_delta.append(abs(loss[i]-loss[i+1]))
            loss_delta = np.array(loss_delta)
            for i in range(loss_delta.shape[0]-1):
                loss_delta_2.append(abs(loss_delta[i]-loss[i+1]))
            loss_delta_2 = np.array(loss_delta_2)

            loss = torch.sum(loss) / sum(is_inlier)
            loss_delta = np.sum(loss_delta)/sum(is_inlier)
            loss_delta_2 = np.sum(loss_delta_2)/sum(is_inlier)

            # print(loss.shape)
            total_val_loss = total_val_loss + loss.item() + loss_delta.item() + loss_delta_2.item()
        print ("Training set baseline:", total_train_loss / len(self.train_loader)
            , " Validation set baseline:", total_val_loss / len(self.val_loader))


    def fit(self, model_dir, save_prefix, plot_path):
        self.loss_fn = torch.nn.L1Loss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        training_loss_avg = []
        validation_loss_avg = []

        n_iter = 100
        save_every_epoch = 10
        start_time = time.time()

        # print a baseline for validation set
        self.count_baseline()

        for epoch in range(n_iter):
            # training
            total_train_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                features = batch[0][0].to(self.device)
                score_pitch = batch[1][0].to(self.device)
                pitch_diff = batch[2][0].float().to(self.device)
                is_inlier = batch[3][0]
                former_note = batch[4][0].to(self.device)
                next_note = batch[5][0].to(self.device)
                former_distance = batch[6][0].to(self.device)
                latter_distance = batch[7][0].to(self.device)

                pitch_diff_pred = self.model(features, score_pitch, former_note, next_note, former_distance, latter_distance).squeeze(1)
                is_inlier = is_inlier.float().squeeze().to(self.device)
                loss = self.loss_fn(pitch_diff_pred, pitch_diff) * is_inlier
                loss_delta = []
                loss_delta_2 = []
                for i in range(loss.shape[0]-1):
                    loss_delta.append(abs(loss[i]-loss[i+1]))
                loss_delta = np.array(loss_delta)
                for i in range(loss_delta.shape[0]-1):
                    loss_delta_2.append(abs(loss_delta[i]-loss[i+1]))
                loss_delta_2 = np.array(loss_delta_2)

                loss = torch.sum(loss) / sum(is_inlier)
                loss_delta = np.sum(loss_delta)/sum(is_inlier)
                loss_delta_2 = np.sum(loss_delta_2)/sum(is_inlier)

                # print(loss.shape)
                total_train_loss = total_train_loss + loss.item() + loss_delta.item() + loss_delta_2.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            training_loss_avg.append(total_train_loss / len(self.train_loader))

            # validation
            with torch.no_grad():
                total_val_loss = 0
                for batch_idx, batch in enumerate(self.val_loader):
                    features = batch[0][0].to(self.device)
                    score_pitch = batch[1][0].to(self.device)
                    pitch_diff = batch[2][0].float().to(self.device)
                    is_inlier = batch[3][0]
                    former_note = batch[4][0].to(self.device)
                    next_note = batch[5][0].to(self.device)
                    former_distance = batch[6][0].to(self.device)
                    latter_distance = batch[7][0].to(self.device)


                    pitch_diff_pred = self.model(features, score_pitch, former_note, next_note, former_distance, latter_distance).squeeze(1)
                
                    is_inlier = is_inlier.float().squeeze().to(self.device)
                    # print(sum(is_inlier))
                    # print(is_inlier.shape)
                    

                    loss = self.loss_fn(pitch_diff_pred, pitch_diff) * is_inlier
                    loss_delta = []
                    loss_delta_2 = []
                    for i in range(loss.shape[0]-1):
                        loss_delta.append(abs(loss[i]-loss[i+1]))
                    loss_delta = np.array(loss_delta)
                    for i in range(loss_delta.shape[0]-1):
                        loss_delta_2.append(abs(loss_delta[i]-loss[i+1]))
                    loss_delta_2 = np.array(loss_delta_2)

                    loss = torch.sum(loss) / sum(is_inlier)
                    loss_delta = np.sum(loss_delta)/sum(is_inlier)
                    loss_delta_2 = np.sum(loss_delta_2)/sum(is_inlier)

                    total_val_loss = total_val_loss + loss.item() + loss_delta.item() + loss_delta_2.item()
                    



            validation_loss_avg.append(total_val_loss / len(self.val_loader))
            
            if epoch % 10 == 0:
                print ("Epoch", epoch, "time passed:", time.time() - start_time, "Training loss:", total_train_loss / len(self.train_loader)
                    , "Validation loss:", total_val_loss / len(self.val_loader))

            if (epoch+1) % save_every_epoch == 0:
                save_dict = self.model.state_dict()
                target_model_path = Path(model_dir) / (save_prefix+'_{}'.format(epoch+1))
                torch.save(save_dict, target_model_path)

            if epoch == (n_iter-1):
                pitch_curve_gt = (score_pitch + pitch_diff).to(self.device)
                pitch_curve_pred = (score_pitch + pitch_diff_pred).to(self.device)
                pitch_curve_pred, pitch_curve_gt, segments = self.do_segment(pitch_curve_pred, pitch_curve_gt, is_inlier, return_idx=True)
                # return pitch_curve_pred, pitch_curve_gt, segments
                # self.plot_pitch_line(pitch_curve_gt, pitch_curve_pred, segments, plot_path_2)




        self.draw_loss(training_loss_avg, validation_loss_avg, plot_path)
        return pitch_curve_pred, pitch_curve_gt, segments
        # self.plot_pitch_line()



    def plot_pitch_line(self, pitch_curve_gt, pitch_curve_pred, segments, plot_path):
        
        frame_duration = 0.01
        plt.rcParams["figure.figsize"] = (segments[-1][1] / 1000.0, 5)

        for i in range(len(pitch_curve_gt) - 1):
            time_index = np.arange(segments[i][0], segments[i][1], 1) * frame_duration
            plt.plot(time_index, pitch_curve_gt[i], color='b', linewidth=1.0, alpha=0.5)
            plt.plot(time_index, pitch_curve_pred[i], color='r', linewidth=0.5)


        for i in range(len(pitch_curve_gt) - 1, len(pitch_curve_gt)):
            time_index = np.arange(segments[i][0], segments[i][1], 1) * frame_duration
            plt.plot(time_index, pitch_curve_gt[i], color='b', label='Groundtruth', linewidth=2.0, alpha=0.5)
            plt.plot(time_index, pitch_curve_pred[i], color='r', label='Prediction', linewidth=1.0)

        plt.xlabel("Time (sec)")
        plt.ylabel("Pitch value")

        plt.legend(loc="best", fontsize=10)

        # plt.show()
        plt.savefig(plot_path, dpi=500)


    def do_segment(self, pitch_diff_pred, pitch_diff_gt, is_inlier, return_idx=False):
        ret_pred = []
        ret_gt = []
        start = 0
        is_segment = 0
        segments = []
        for i in range(len(is_inlier)):
            if is_inlier[i] == 0 and is_segment == 1:
                segments.append([start, i])
                ret_pred.append(pitch_diff_pred[start:i].to("cpu"))
                ret_gt.append(pitch_diff_gt[start:i].to("cpu"))
                is_segment = 0

            elif is_inlier[i] == 1 and is_segment == 0:
                is_segment = 1
                start = i

        if is_segment == 1:
            ret_pred.append(pitch_diff_pred[start:].to("cpu"))
            ret_gt.append(pitch_diff_gt[start:].to("cpu"))
            segments.append([start, len(is_inlier)])

        if return_idx == True:
            return ret_pred, ret_gt, segments
        else:
            return ret_pred, ret_gt

