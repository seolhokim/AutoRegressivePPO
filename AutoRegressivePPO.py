import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#
class AutoRegressivePPO(nn.Module):
    def __init__(self):
        super(AutoRegressivePPO,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.encoder = nn.Sequential(
                        nn.Conv2d(3,64,3, stride= 1 , padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2), #64 * 64
                        nn.Conv2d(64,128,3, stride= 1 , padding = 1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2), # 32 * 32
                        nn.Conv2d(128,256,5, stride= 1 , padding = 2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2,stride=2), # 16 * 16
                        nnFlatten(),
                        nn.Linear(256 * 16 * 16,1024)
                        nn.ReLU()
                        )
        self.value = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(1024,1)
                        )
        
        self.action_x = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(1024,action_x_dim),
                            nn.Softmax(-1)
                            )
        self.action_x_embedding = nn.Embedding(action_x_dim,32)
        
        self.action_y = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(1024,32),
                            )
        self.action_y_last = nn.Linear(32 + action_x_dim,action_y_dim)
        
    def forward(self,x,action_sequence = []):
        encoding = self.encoder(x)

        value = self.value(encoding)

        action_x = self.action_x(encoding)

        action_x_categorical = Categorical(action_x)
        if len(action_sequence) > 0:
            action_x_sample = action_sequence[0]
        else :
            action_x_sample = action_x_categorical.sample()
        action_x_log_prob = action_x_categorical.log_prob(action_x_sample) 

        action_x = self.action_x_embedding(action_x_sample)

        action_y = self.action_y(encoding)
        action_y = torch.cat([action_x,action_y],-1)
        action_y = F.relu(action_y)
        action_y = self.action_y_last(action_y)
        action_y = F.softmax(action_y,-1)

        action_y_categorical = Categorical(action_y)
        if len(action_sequence) > 0:
            action_y_sample = action_sequence[1]
        else : 
            action_y_sample = action_y_categorical.sample()
        action_y_log_prob = action_y_categorical.log_prob(action_y_sample)

        return [action_x_sample,action_y_sample],\
                [action_x_log_prob,action_y_log_prob],value