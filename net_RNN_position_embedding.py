import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.RNN(488, 100)
        self.rnn2 = nn.RNN(100,1)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'


    def compute_pe(self, y, dim):
        output = torch.zeros((y.shape[-1], dim), dtype=torch.float)
        for i in range(output.shape[1]):
            if i % 2 == 0:
                output[:,i] = torch.sin(y / np.power(100000, (float(i)/dim)))
            else:
                output[:,i] = torch.cos(y / np.power(100000, (float(i-1)/dim)))

        return output

    def forward(self, features, score_pitch, former_note, next_note, former_distance, latter_distance):
        
        features = features.float()
        score_pitch = self.compute_pe(score_pitch, dim=100).to(self.device)
        former_note = former_note.float()
        next_note =  next_note.float()
        former_distance = former_distance.float()
        latter_distance = latter_distance.float()

        out = torch.cat((features, score_pitch, former_note.unsqueeze(1), next_note.unsqueeze(1), 
                            former_distance.unsqueeze(1), latter_distance.unsqueeze(1)), dim=1)
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