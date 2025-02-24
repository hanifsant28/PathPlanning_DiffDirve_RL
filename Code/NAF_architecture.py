import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

class NAF_DQNN(nn.Module):
    def __init__(self, hidden_size, action_size, state_size, lidar_size, lidar_batch_num, max_action, device):
        super().__init__()
        self.lidar_size = lidar_size
        self.lidar_batch_num = lidar_batch_num
        self.device = device

        self.action_dims = action_size
        self.max_action = torch.tensor(max_action).to(self.device)
        
        # base network
        self.net = nn.Sequential(
            nn.Linear(state_size + lidar_batch_num, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(self.device)  # Ensure the network is on the correct device

        # action policy
        self.linear_mu = nn.Linear(hidden_size, action_size).to(self.device)  # Ensure the layer is on the correct device

        # state value
        self.linear_value = nn.Linear(hidden_size, 1).to(self.device)  # Ensure the layer is on the correct device

        # L matrix
        self.linear_matrix = nn.Linear(hidden_size, int(action_size * (action_size + 1) / 2)).to(self.device)  # Ensure the layer is on the correct device

    @torch.no_grad()
    def mu(self, input):
        x, lidar = input
        
        nn_lidar = self.preprocess_lidar(lidar)
        x = torch.cat((x, nn_lidar), dim=-1)  # Concatenate along the last dimension
        
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.net(x)
        x = self.linear_mu(x)
        x = torch.tanh(x) * self.max_action
        return x

    @torch.no_grad()
    def value(self, input):
        x, lidar = input
        
        nn_lidar = self.preprocess_lidar(lidar)
        x = torch.cat((x, nn_lidar), dim=-1)  # Concatenate along the last dimension
        
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.net(x)
        x = self.linear_value(x)
        return x

    def preprocess_lidar(self, lidar_data):
        if lidar_data.dim() == 1:
            lidar_data = lidar_data.unsqueeze(0)  # Add batch dimension if missing

        lidar_data = lidar_data.to(self.device)  # Ensure lidar_data is on the correct device

        data_length = int(self.lidar_size / self.lidar_batch_num)
        nn_data = []

        for i in range(self.lidar_batch_num):
            if i + 1 == self.lidar_batch_num:
                # If it is the last batch, take all the rest of the rays into the same batch
                batch_segment = lidar_data[:, i * data_length:]
            else:
                batch_segment = lidar_data[:, i * data_length : (i + 1) * data_length]
            
            # Find the minimum value in each segment
            min_values = torch.min(batch_segment, dim=1).values
            nn_data.append(min_values)

        # Stack the minimum values to form a tensor of shape [batch_size, lidar_batch_num]
        nn_data = torch.stack(nn_data, dim=1).to(self.device)
        return nn_data

    def forward(self, input, a):
        x, lidar = input
        
        nn_lidar = self.preprocess_lidar(lidar)
        x = torch.cat((x, nn_lidar), dim=-1)  # Concatenate along the last dimension
        
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.net(x)
        mu = torch.tanh(self.linear_mu(x)) * self.max_action
        value = self.linear_value(x)

        # P(X)
        matrix = torch.tanh(self.linear_matrix(x))

        L = torch.zeros((x.shape[0], self.action_dims, self.action_dims)).to(self.device)
        tril_indices = torch.tril_indices(row=self.action_dims, col=self.action_dims).to(self.device)
        L[:, tril_indices[0], tril_indices[1]] = matrix
        P = L @ L.transpose(2, 1)

        u_mu = (a - mu).unsqueeze(dim=1).to(self.device)
        u_mu_t = u_mu.transpose(1, 2).to(self.device)

        adv = -0.5 * u_mu @ P @ u_mu_t
        adv = adv.squeeze(dim=-1)
        return value + adv

def noisy_policy(state, net, epsilon, device):
    amin = torch.tensor([0, 0], dtype=torch.float32).to(device)
    amax = torch.tensor([1, 1], dtype=torch.float32).to(device)

    mu = net.mu(state)
    mu = mu + torch.normal(0, epsilon, mu.shape).to(device)
    action = torch.clamp(mu, amin, amax)
    return action.squeeze().cpu().tolist()  # Squeeze, move to CPU, and convert to list before returning