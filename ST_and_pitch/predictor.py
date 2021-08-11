import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
import time
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import Counter
import numpy as np

import sys
import os

from net import EffNetb0
import math
import statistics
from data_utils import AudioDataset
from evaluate import MirEval

FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)
# FRAME_LENGTH = 0.01

class EffNetPredictor:
    def __init__(self, device= "cuda:0", model_path=None):
        """
        Params:
        model_path: Optional pretrained model file
        """
        # Initialize model
        self.device = device


        if model_path is not None:
            self.model = EffNetb0().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location= self.device))
            print('Model read from {}.'.format(model_path))

        else:
            # svs_path = "../multitask/model/0128_e_500"
            svs_path = "../multitask/model/0112_e_200"
            
            self.model = EffNetb0(svs_path=svs_path).to(self.device)

        print('Predictor initialized.')

    def test_and_eval(self):
        my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            for batch_idx, batch in enumerate(self.test_set_loader):
                input_tensor = batch[0].to(self.device)
                song_ids = batch[2]

                # print (input_tensor.shape)

                result_tuple = self.model(input_tensor)
                # _, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, all_result = result_tuple
                # on_off_logits = result_tuple[0]
                on_off_logits = result_tuple[1]
                # on_off_logits = F.softmax(on_off_logits, dim=2)
                # pitch_octave_logits = result_tuple[1]
                # pitch_class_logits = result_tuple[2]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                # onset_logits = result_tuple[0]
                # offset_logits = result_tuple[1]
                # pitch_octave_logits = result_tuple[2]
                # pitch_class_logits = result_tuple[3]
                # onset_d_logits = result_tuple[4]

                onset_logits = on_off_logits[:, :, 1]
                offset_logits = on_off_logits[:, :, 2]

                # onset_probs, offset_probs, onset_d_probs = (torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                #     , torch.sigmoid(onset_d_logits).cpu())
                onset_probs, offset_probs = (onset_logits.cpu(), offset_logits.cpu())
                pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    for i in range(len(onset_probs[bid])):
                        frame_info = (onset_probs[bid][i], offset_probs[bid][i], torch.argmax(pitch_octave_logits[bid][i])
                            , torch.argmax(pitch_class_logits[bid][i]).item())
                        song_frames_table.setdefault(song_id, [])
                        song_frames_table[song_id].append(frame_info)

            # Parse frame info into output format for every song

            onset_thres_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            best_con = 0.0
            best_result = None
            best_result10 = None
            best_con_thres = None
            for onset_candidate in onset_thres_set:
                results = {}
                for song_id, frame_info in song_frames_table.items():
                    # print (song_id)
                    results[song_id] = self._parse_frame_info(frame_info, onset_thres=onset_candidate, offset_thres=0.5)

                self.eval_class.add_tr_tuple_and_prepare(results)
                eval_result = self.eval_class.accuracy(0.05, method="traditional", print_result=False)
                if eval_result[8] >= best_con:
                    best_con = eval_result[8]
                    best_result = list(eval_result)
                    best_con_thres = onset_candidate

                    eval_result10 = self.eval_class.accuracy(0.1, method="traditional", print_result=False)
                    best_result10 = list(eval_result10)
                    
            print ("Now best result(onset_tol=0.05): onset threshold =", best_con_thres)
            print("         Precision Recall F1-score")
            print("COnPOff  %f %f %f" % (best_result[0], best_result[1], best_result[2]))
            print("COnP     %f %f %f" % (best_result[3], best_result[4], best_result[5]))
            print("COn      %f %f %f" % (best_result[6], best_result[7], best_result[8]))
            print ("gt note num:", best_result[9], "tr note num:", best_result[10])
            print ("*When onset_tol=0.1:")
            print("COnPOff  %f %f %f" % (best_result10[0], best_result10[1], best_result10[2]))
            print("COnP     %f %f %f" % (best_result10[3], best_result10[4], best_result10[5]))
            print("COn      %f %f %f" % (best_result10[6], best_result10[7], best_result10[8]))

        return best_result, best_result10

    def print_onset_num(self, cur_loader):
        total = 0
        onset_true = 0
        onset_d_true = 0
        for batch_idx, batch in enumerate(cur_loader):
            onset_prob = batch[2][:, :, 0]
            onset_dilated = batch[2][:, :, 4]

            # print (onset_prob.shape)
            for cur_instance in onset_prob:
                for element in cur_instance:
                    total = total + 1
                    if int(element) == 1:
                        onset_true = onset_true + 1

            for cur_instance in onset_dilated:
                for element in cur_instance:
                    if int(element) == 1:
                        onset_d_true = onset_d_true + 1

        print (onset_true, onset_d_true, total)

    def count_loss(self, batch, model_output, total_split_loss, use_ctc=True):
        # svs_gt = batch[0][:, -3:, :, :]
        onset_prob = batch[3][:, :, 0].float().to(self.device)
        offset_prob = batch[3][:, :, 1].float().to(self.device)
        pitch_octave = batch[3][:, :, 2].long().to(self.device)
        pitch_class = batch[3][:, :, 3].long().to(self.device)

        # onset_dilated = batch[2][:, :, 4].float().to(self.device)
        on_off_seq = []
        # print (batch[4][0])
        # print (batch[4][0])
        for i in range(len(batch[4][0])):
            on_off_seq.append(1)
            on_off_seq.append(int(batch[4][0][i][2])-36+3)
            # on_off_seq.append(3)
            on_off_seq.append(2)
        on_off_seq = torch.tensor([on_off_seq,])

        _, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, all_result = model_output
        # on_off_logits_sm = F.softmax(on_off_logits, dim=2)
        
        on_off_logits_local_m = F.max_pool2d(on_off_logits_sm, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        # on_off_logits_sm2 = F.max_pool2d(on_off_logits, kernel_size=(5, 1), stride=(1, 1)
        #     , padding=(2, 0)).unsqueeze(0) - FRAME_LENGTH * 40.0

        on_off_logits_sm_neg = -on_off_logits_sm
        on_off_logits_local_min = F.max_pool2d(on_off_logits_sm_neg, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))

        onset_logits = on_off_logits_sm[:, :, 1]
        offset_logits = on_off_logits_sm[:, :, 2]

        on_weight = batch[2][0][0:1].float().to(self.device)
        off_weight = batch[2][0][1:2].float().to(self.device)
        onset_gt = batch[5][0][0:1]
        offset_gt = batch[5][0][1:2]

        # print (list(onset_gt[0]))

        onset_gt = onset_gt.float().to(self.device)
        offset_gt = offset_gt.float().to(self.device)

        selected_on = torch.nonzero(on_off_logits_sm[:,:,1] == on_off_logits_local_m[:,:,1], as_tuple=True)
        selected_off = torch.nonzero(on_off_logits_sm[:,:,2] == on_off_logits_local_m[:,:,2], as_tuple=True)

        selected_on_neg = torch.nonzero(on_off_logits_sm_neg[:,:,1] == on_off_logits_local_min[:,:,1], as_tuple=True)
        selected_off_neg = torch.nonzero(on_off_logits_sm_neg[:,:,2] == on_off_logits_local_min[:,:,2], as_tuple=True)
        # print (len(onset_logits))

        if use_ctc == True:
            cur_on_weight_pos = torch.sum(on_weight[selected_on])
            cur_on_weight_neg = torch.sum(on_weight[selected_on_neg])
            split_train_loss0_pos = self.onset_criterion(onset_logits[selected_on], onset_gt[selected_on])
            split_train_loss0_neg = self.onset_criterion(onset_logits[selected_on_neg], onset_gt[selected_on_neg])
            split_train_loss0 = (torch.dot(split_train_loss0_pos, on_weight[selected_on]) / cur_on_weight_pos
                                + torch.dot(split_train_loss0_neg, on_weight[selected_on_neg]) / cur_on_weight_neg)
                        
            cur_off_weight_pos = torch.sum(off_weight[selected_off])
            cur_off_weight_neg = torch.sum(off_weight[selected_off_neg])

            split_train_loss1_pos = self.offset_criterion(offset_logits[selected_off], offset_gt[selected_off])
            split_train_loss1_neg = self.offset_criterion(offset_logits[selected_off_neg], offset_gt[selected_off_neg])

            split_train_loss1 = (torch.dot(split_train_loss1_pos, off_weight[selected_off]) / cur_off_weight_pos
                                + torch.dot(split_train_loss1_neg, off_weight[selected_off_neg]) / cur_off_weight_neg)

        else:
            # split_train_loss0 = torch.sum(self.onset_criterion(onset_logits, onset_prob)) / len(onset_logits[0])
            # print (self.onset_criterion(onset_logits, onset_prob).shape)
            # print (on_weight.shape)
            split_train_loss0 = torch.dot(self.onset_criterion(onset_logits, onset_prob)[0], on_weight[0]) / torch.sum(on_weight)
            # split_train_loss1 = torch.sum(self.offset_criterion(offset_logits, offset_prob)) / len(offset_logits[0])
            split_train_loss1 = torch.dot(self.offset_criterion(offset_logits, offset_prob)[0], off_weight[0]) / torch.sum(off_weight)


        split_train_loss2 = self.octave_criterion(pitch_octave_logits.permute(0, 2, 1), pitch_octave)
        split_train_loss3 = self.pitch_criterion(pitch_class_logits.permute(0, 2, 1), pitch_class)

        # print (split_train_loss0.item(), len(onset_gt[0]))
        # print (on_weight[selected_on].shape)

        # split_train_loss4 = self.onset_dilated_criterion(onset_d_logits, onset_dilated)
        # on_off_ctc_logits = F.log_softmax(on_off_logits, dim=2).permute(1, 0, 2)

        on_off_ctc_logits = all_result.permute(1, 0, 2)

        if use_ctc == True:
            split_train_loss4 = self.on_off_criterion(on_off_ctc_logits, on_off_seq
                , (on_off_ctc_logits.shape[0],), (on_off_seq.shape[1],))
        else:
            split_train_loss4 = torch.zeros((1,)).to(self.device)
        # svs_train_loss = self.svs_loss(svs_result, svs_gt)
        
        total_split_loss[0] = total_split_loss[0] + split_train_loss0.item()
        total_split_loss[1] = total_split_loss[1] + split_train_loss1.item()
        total_split_loss[2] = total_split_loss[2] + split_train_loss2.item()
        total_split_loss[3] = total_split_loss[3] + split_train_loss3.item()
        total_split_loss[4] = total_split_loss[4] + split_train_loss4.item()
        # total_split_loss[5] = total_split_loss[5] + svs_train_loss.item() 

        # unweighted_loss = split_train_loss0 + split_train_loss1 + split_train_loss2 + split_train_loss3 + split_train_loss4 + svs_train_loss
        loss = split_train_loss0 + split_train_loss1 + split_train_loss2 + split_train_loss3 + split_train_loss4
        # loss = split_train_loss0 + split_train_loss1 + split_train_loss2 + split_train_loss3 + split_train_loss4 + svs_train_loss * svs_weight
        return loss


    def fit(self, model_dir, **training_args):
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Set training params
        self.batch_size = training_args['batch_size']
        self.valid_batch_size = training_args['valid_batch_size']
        self.epoch = training_args['epoch']
        self.lr = training_args['lr']
        self.save_every_epoch = training_args['save_every_epoch']
        self.plot_path = training_args['plot_path']

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # onset:onset_d:total_frame = 129521:905069:5267200 in training set
        # onset:onset_d:total_frame = 325688:2275820:10534400 in augmented training set
        # augmented: (32.34, 4.63)

        # cmedia dataset: 383127:2582955:9452800 (24.673, 3.660)

        # cmedia new dataset: 241082:1626540:5696000
        # self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([24.673,], device=self.device))
        # self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([32.34,], device=self.device))
        # , reduction='none'
        # self.onset_criterion = nn.MSELoss()
        # self.offset_criterion = nn.MSELoss()
        # pos_weight=torch.tensor([5267200.0/129521.0,], device=self.device)
        # pos_weight=torch.tensor([5.0,], device=self.device)
        self.onset_criterion = nn.BCELoss(reduction='none')
        # self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5696000.0/241082.0,], device=self.device))
        self.offset_criterion = nn.BCELoss(reduction='none')
        # self.onset_dilated_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.66,], device=self.device))
        # self.onset_dilated_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.63,], device=self.device))

        # self.onset_dilated_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5696000.0/1626540.0], device=self.device))
        self.on_off_criterion = nn.CTCLoss(blank=0)

        self.octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)
        self.svs_loss = nn.L1Loss()

        # Read the datasets
        print('Reading datasets...')
        print ('cur time: %.6f' %(time.time()))
        
        self.test_set = None
        with open("val_set_0314_voc_and_mix.pkl", 'rb') as f:
        # # with open("val_set_0128_12testset.pkl", 'rb') as f:
            self.test_set = pickle.load(f)


        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.valid_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.test_set_loader = DataLoader(
            self.test_set,
            batch_size=2,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        self.eval_class = MirEval()
        # self.eval_class.add_gt("json/MIR-cmedia480_test.json")
        self.eval_class.add_gt("../ST/json/MIR-ST500_corrected_1005.json")

        start_time = time.time()
        training_loss_list = []
        valid_loss_list = []
        split_loss_list = []
        valid_split_loss_list = []
        result_index_list = []

        # Start training
        print('Start training...')
        print ('cur time: %.6f' %(time.time()))
        self.iters_per_epoch = len(self.train_loader)
        print (self.iters_per_epoch)

        # result_index = self.test_and_eval()
        for epoch in range(1, self.epoch + 1):
            self.model.train()
            # print (self.model.svs.conv1.do.training)
            # Run iterations
            total_training_loss = 0
            total_split_loss = np.zeros(5)
            # svs_weight = 100.0 * math.exp(-epoch/20.0)

            # svs_weight = 200.0

            # self.print_onset_num(self.train_loader)

            for batch_idx, batch in enumerate(self.train_loader):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                # input_2 = batch[1].to(self.device)
                loss = 0
                length = len(input_tensor)
                
                self.optimizer.zero_grad()

                model_output = self.model(input_tensor)
                # onset_logits, offset_logits, pitch_ans = self.model(input_tensor, input_2)

                if epoch <= 5:
                # if True:
                    loss = self.count_loss(batch, model_output, total_split_loss, use_ctc=False)
                else:
                    loss = self.count_loss(batch, model_output, total_split_loss, use_ctc=True)
                # loss, unweighted_loss = self.count_loss(batch, model_output, total_split_loss)
                
                loss.backward()
                self.optimizer.step()
                # total_training_loss += unweighted_loss.item()
                total_training_loss += loss.item()

                if batch_idx % 200 == 0 and batch_idx != 0:
                    print (epoch, batch_idx, "time:", time.time()-start_time, "loss:", total_training_loss / (batch_idx+1))


            if epoch % self.save_every_epoch == 0:
                # Perform validation
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    split_val_loss = np.zeros(5)
                    for batch_idx, batch in enumerate(self.valid_loader):

                        input_tensor = batch[0].to(self.device)
                        # input_2 = batch[1].to(self.device)

                        model_output = self.model(input_tensor)

                        if epoch <= 5:
                        # if True:
                            loss = self.count_loss(batch, model_output, split_val_loss, use_ctc=False)
                        else:
                            loss = self.count_loss(batch, model_output, split_val_loss, use_ctc=True)

                        # loss = self.count_loss(batch, model_output, split_val_loss)
                        # loss, unweighted_loss = self.count_loss(batch, model_output, split_val_loss)

                        total_valid_loss += loss.item()
                        # total_valid_loss += unweighted_loss.item()

                # Save model
                save_dict = self.model.state_dict()
                target_model_path = Path(self.model_dir) / (training_args['save_prefix']+'_{}'.format(epoch))
                # target_model_path = Path(self.model_dir) / str(training_args['save_prefix'])+'_{}'.format(epoch)
                torch.save(save_dict, target_model_path)

                # Save loss list
                training_loss_list.append((epoch, total_training_loss/len(self.train_loader)))
                valid_loss_list.append((epoch, total_valid_loss/len(self.valid_loader)))
                split_loss_list.append((epoch, total_split_loss/len(self.train_loader)))
                valid_split_loss_list.append((epoch, split_val_loss/len(self.valid_loader)))

                # Epoch statistics
                print(
                    '| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        self.epoch,
                        training_loss_list[-1][1],
                        valid_loss_list[-1][1],
                        time.time()-start_time))

                print('split train loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} on_off_ctc {:.4f}'.format(
                        total_split_loss[0]/len(self.train_loader),
                        total_split_loss[1]/len(self.train_loader),
                        total_split_loss[2]/len(self.train_loader),
                        total_split_loss[3]/len(self.train_loader),
                        total_split_loss[4]/len(self.train_loader)
                    )
                )
                print('split val loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} on_off_ctc {:.4f}'.format(
                        split_val_loss[0]/len(self.valid_loader),
                        split_val_loss[1]/len(self.valid_loader),
                        split_val_loss[2]/len(self.valid_loader),
                        split_val_loss[3]/len(self.valid_loader),
                        split_val_loss[4]/len(self.valid_loader)
                    )
                )

                # print('split train loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} onset_d {:.4f} svs {:.4f}'.format(
                #         total_split_loss[0]/len(self.train_loader),
                #         total_split_loss[1]/len(self.train_loader),
                #         total_split_loss[2]/len(self.train_loader),
                #         total_split_loss[3]/len(self.train_loader),
                #         total_split_loss[4]/len(self.train_loader),
                #         total_split_loss[5]/len(self.train_loader)
                #     )
                # )
                # print('split val loss: onset {:.4f} offset {:.4f} pitch octave {:.4f} pitch chroma {:.4f} onset_d {:.4f} svs {:.4f}'.format(
                #         split_val_loss[0]/len(self.valid_loader),
                #         split_val_loss[1]/len(self.valid_loader),
                #         split_val_loss[2]/len(self.valid_loader),
                #         split_val_loss[3]/len(self.valid_loader),
                #         split_val_loss[4]/len(self.valid_loader),
                #         split_val_loss[5]/len(self.valid_loader)
                #     )
                # )

                if epoch % 5 == 0 or epoch == 1:
                    result_index5, result_index10 = self.test_and_eval()
                    result_index_list.append([epoch, result_index5, result_index10])

        # Save loss to file
        with open(self.plot_path, 'wb') as f:
            pickle.dump({'train': training_loss_list, 'valid': valid_loss_list, 'train_split':split_loss_list, 'valid_split':valid_split_loss_list
                , 'result_index': result_index_list}, f)
            # pickle.dump({'train': training_loss_list, 'valid': valid_loss_list, 'train_split':split_loss_list, 'valid_split':valid_split_loss_list}, f)

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))

    def _parse_frame_info(self, frame_info, onset_thres, offset_thres):
        """Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format."""

        result = []
        current_onset = None
        # chroma_counter = Counter()
        # octave_counter = Counter()
        # pitch_counter = Counter()

        pitch_counter = []

        last_onset = 0.0
        # onset_d_thres = onset_thres
        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        # onset_d = np.array([frame_info[i][4] for i in range(len(frame_info))])

        local_max_size = 3
        current_time = 0.0

        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            # print (i*0.032, info[0])

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            # if info[0] >= onset_thres and info[4] >= onset_d_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time
                    last_onset = info[0] - onset_thres

                else:
                # elif info[0] >= onset_thres:
                    # If current_onset exists, make this onset a offset and the next current_onset
                    # result_pitch = chroma_counter.most_common(1)[0][0] + octave_counter.most_common(1)[0][0] * 12 + 36
                    # result.append([current_onset, current_time, result_pitch])

                    # if sum(pitch_counter.values()) > 0:
                    if len(pitch_counter) > 0:
                        # result.append([current_onset, current_time, pitch_counter.most_common(1)[0][0] + 36])
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                        

                    # if chroma_counter.most_common(1)[0][0] != 49:
                    #     result.append([current_onset, current_time, chroma_counter.most_common(1)[0][0] + 36])
                    # elif len(chroma_counter.most_common(2)) == 2:
                    #     result.append([current_onset, current_time, chroma_counter.most_common(2)[1][0] + 36])
                    current_onset = current_time
                    last_onset = info[0] - onset_thres
                    # chroma_counter.clear()
                    # octave_counter.clear()
                    # pitch_counter.clear()
                    pitch_counter = []

            elif info[1] >= offset_thres:  # If is offset
                if current_onset is not None:
                    # result_pitch = chroma_counter.most_common(1)[0][0] + octave_counter.most_common(1)[0][0] * 12 + 36
                    # result.append([current_onset, current_time, result_pitch])

                    # if sum(pitch_counter.values()) > 0:
                    if len(pitch_counter) > 0:
                        # result.append([current_onset, current_time, pitch_counter.most_common(1)[0][0] + 36])
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

                    # if chroma_counter.most_common(1)[0][0] != 49:
                    #     result.append([current_onset, current_time, chroma_counter.most_common(1)[0][0] + 36])
                    # elif len(chroma_counter.most_common(2)) == 2:
                    #     result.append([current_onset, current_time, chroma_counter.most_common(2)[1][0] + 36])
                    current_onset = None
                    # chroma_counter.clear()
                    # octave_counter.clear()
                    # pitch_counter.clear()
                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                # chroma_counter[int(info[3])] += 1
                # octave_counter[int(info[2])] += 1

                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                    # pitch_counter[final_pitch] += 1
                    pitch_counter.append(final_pitch)

                # for j in range(12):
                #     chroma_counter[j] += info[5][j] * (1.0-info[1])
                # for j in range(4):
                #     octave_counter[j] += info[4][j] * (1.0-info[1])

        # print (current_onset)
        if current_onset is not None:
            # result_pitch = chroma_counter.most_common(1)[0][0] + octave_counter.most_common(1)[0][0] * 12 + 36
            # result.append([current_onset, current_time, result_pitch])
            
            # if sum(pitch_counter.values()) > 0:
            if len(pitch_counter) > 0:
                # result.append([current_onset, current_time, pitch_counter.most_common(1)[0][0] + 36])
                result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])

            # if chroma_counter.most_common(1)[0][0] != 49:
            #     result.append([current_onset, current_time, chroma_counter.most_common(1)[0][0] + 36])
            # elif len(chroma_counter.most_common(2)) == 2:
            #     result.append([current_onset, current_time, chroma_counter.most_common(2)[1][0] + 36])
            current_onset = None
            # chroma_counter.clear()
            # octave_counter.clear()
            # pitch_counter.clear()
            pitch_counter = []

        return result

    def predict(self, test_dataset, results= {}, show_tqdm= True, onset_thres=0.1, offset_thres=0.5):
        """Predict results for a given test dataset."""
        # Setup params and dataloader
        batch_size = 1
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        # Start predicting
        my_sm = torch.nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            # print (self.model.svs.conv1.do.training)
            song_frames_table = {}
            raw_data = {}

            if show_tqdm == True:
                print('Forwarding model...')
                for batch_idx, batch in enumerate(tqdm(test_loader)):
                    # Parse batch data
                    # input_tensor = batch[0].permute(0, 2, 1).unsqueeze(1).to(self.device)
                    input_tensor = batch[0].to(self.device)

                    song_ids = batch[1]

                    result_tuple = self.model(input_tensor)
                    on_off_logits = result_tuple[1]
                    pitch_octave_logits = result_tuple[2]
                    pitch_class_logits = result_tuple[3]

                    onset_logits = on_off_logits[:, :, 1]
                    offset_logits = on_off_logits[:, :, 2]

                    # onset_logits, offset_logits, pitch_ans = self.model(input_tensor, input_2)
                    # print (pitch_ans)
                    onset_probs, offset_probs = (onset_logits.cpu(), offset_logits.cpu())
                    # print (onset_probs[0][2000:2100].numpy())
                    # print (offset_probs[0][2000:2100].numpy())
                    # onset_probs, offset_probs, onset_d_probs = (torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                    #     , torch.sigmoid(onset_d_logits).cpu())
                    pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()


                    # Collect frames for corresponding songs
                    for bid, song_id in enumerate(song_ids):
                        for i in range(len(onset_probs[bid])):
                            # frame_info = (onset_probs[bid][i], offset_probs[bid][i], torch.argmax(pitch_octave_logits[bid][i])
                            #     , torch.argmax(pitch_class_logits[bid][i]).item(), onset_d_probs[bid][i])
                            # frame_info = (onset_probs[bid], offset_probs[bid], pitch_ans[bid])
                            frame_info = (onset_probs[bid][i], offset_probs[bid][i], torch.argmax(pitch_octave_logits[bid][i])
                            , torch.argmax(pitch_class_logits[bid][i]).item())

                            song_frames_table.setdefault(song_id, [])
                            song_frames_table[song_id].append(frame_info)
                        
            # else:
            #     # print('Forwarding model...')
            #     for batch_idx, batch in enumerate(test_loader):
            #         input_tensor = batch[0].to(self.device)
            #         song_ids = batch[1]

            #         onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

            #         onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
            #         pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

            #         # for i in range(0, 1000, 100):
            #         #     print (offset_probs[i:i+100])

            #         # Collect frames for corresponding songs
            #         for bid, song_id in enumerate(song_ids):
            #             frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_octave_logits[bid])
            #                 , torch.argmax(pitch_class_logits[bid]).item(), my_sm(pitch_octave_logits[bid]).numpy(), my_sm(pitch_class_logits[bid]).numpy())
            #             song_frames_table.setdefault(song_id, [])
            #             song_frames_table[song_id].append(frame_info)

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                # print (song_id)
                results[song_id] = self._parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
                # results[song_id] = self._parse_pitch(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
                # print (results[song_id])

        return results
