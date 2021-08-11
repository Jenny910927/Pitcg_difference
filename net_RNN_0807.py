import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.RNN(389, 100)
        self.rnn2 = nn.RNN(100,1)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'



    def forward(self, features, score_pitch, former_note, next_note, former_distance, latter_distance):
        
        features = features.float()
        score_pitch = score_pitch.float()
        former_note = former_note.float()
        next_note =  next_note.float()
        former_distance = former_distance.float()
        latter_distance = latter_distance.float()

        # print (features.shape, score_pitch.shape, former_note.shape,  next_note.shape, former_distance.shape, latter_distance.shape)
        out = torch.cat((features, score_pitch.unsqueeze(1), former_note.unsqueeze(1), 
                        next_note.unsqueeze(1), former_distance.unsqueeze(1), latter_distance.unsqueeze(1)), dim=1)
        # print(out.shape[0])
        
        n = out.shape[0]
        
        
        out = out.unsqueeze(1)
        out, _ = self.rnn1(out)
        out = F.tanh(out)
        out, _ = self.rnn2(out)
        
        # print("out.shape: ", out.shape)
        out = out.squeeze(1)
        # print("out.shape: ", out.shape)


        return out