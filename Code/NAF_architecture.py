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
    def __init__(self, hidden_size, action_size, state_size, max_action, device):
        super().__init__()
        self.device = device

        self.action_dims = action_size
        self.max_action = torch.tensor(max_action).to(self.device)
        
        # base network
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ).to(self.device)  # Ensure the network is on the correct device

        # action policy
        self.linear_mu = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, action_size)).to(self.device)  # Ensure the layer is on the correct device

        # state value
        self.linear_value = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, 1)).to(self.device)  # Ensure the layer is on the correct device

        # L matrix
        self.linear_matrix = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, int(action_size * (action_size + 1) / 2))).to(self.device)  # Ensure the layer is on the correct device

    @torch.no_grad()
    def mu(self, input):
        x = input
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.net(x)
        x = self.linear_mu(x)
        x = torch.tanh(x) * self.max_action
        return x

    @torch.no_grad()
    def value(self, input):
        x = input
        x = x.to(self.device)  # Ensure x is on the correct device
        x = self.net(x)
        x = self.linear_value(x)
        return x


    def forward(self, input, a):
        x = input
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


def noisy_policy(state, net, epsilon, device, max_speed):
    amin = torch.tensor([0, 0], dtype=torch.float32).to(device)
    amax = torch.tensor([max_speed, max_speed], dtype=torch.float32).to(device)

    mu = net.mu(state)
    mu = mu + torch.normal(0, epsilon, mu.shape).to(device)
    action = torch.clamp(mu, amin, amax)
    return action.squeeze().cpu().tolist()  # Squeeze, move to CPU, and convert to list before returning