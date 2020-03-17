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


class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()
        self.model = AutoRegressivePPO()
        
        self.memory = []
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)
        
    def get_action(self,x,action_sequence = []):
        action, prob,value = self.model(x,action_sequence)
        return action,prob
    
    def get_value(self,x):
        return self.model(x)[2]
    
    def put_data(self,data):
        self.memory.append(data)
        
    def make_batch(self):
        state_list, action_x_list, action_y_list,\
        reward_list, next_state_list, action_x_prob_list,\
        action_y_prob_list, done_list = [],[],[],[],[],[],[],[]
        for data in self.memory:
            state,action_x,action_y,reward,next_state,action_x_prob,action_y_prob,done = data
            state_list.append(state)
            action_x_list.append([action_x])
            action_y_list.append([action_y])
            reward_list.append([reward])
            next_state_list.append(next_state)
            action_x_prob_list.append([action_x_prob])
            action_y_prob_list.append([action_y_prob])
            done_mask = 0 if done else 1
            done_list.append([done_mask])
        self.memory = []
        
        s,action_x,action_y,r,next_s,action_x_prob,action_y_prob, done_mask =\
                                        torch.tensor(state_list,dtype=torch.float),\
                                        torch.tensor(action_x), torch.tensor(action_y),\
                                        torch.tensor(reward_list),\
                                        torch.tensor(next_state_list,dtype=torch.float),\
                                        torch.tensor(action_x_prob_list,dtype = torch.float),\
                                        torch.tensor(action_y_prob_list,dtype = torch.float),\
                                        torch.tensor(done_list,dtype = torch.float)
                                        
        return s,action_x,action_y,r,next_s,action_x_prob,action_y_prob, done_mask
    
    def train(self):
        state, action_x,action_y, reward, next_state,action_x_prob,\
        action_y_prob, done_mask = self.make_batch()
        for i in range(K_epoch):
            td_error = reward + gamma * self.get_value(next_state) * done_mask
            delta = td_error - self.get_value(state)
            delta = delta.detach().numpy()
            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list,dtype = torch.float)

            
            
            now_action,now_action_prob = self.get_action(state,[action_x[0], action_y[0]])

            now_action_x_type_prob = now_action_prob[0]
            now_action_y_type_prob = now_action_prob[1]
            
            now_action_prob = now_action_x_type_prob + now_action_y_type_prob
            action_prob = action_x_prob + action_x_prob
            
            ratio = torch.exp((now_action_prob) - (action_prob[0]))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio , 1-eps_clip, 1 + eps_clip) * advantage
            loss = - torch.min(surr1,surr2) + F.smooth_l1_loss(self.get_value(state),td_error.detach())
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()