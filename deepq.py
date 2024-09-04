# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:11:36 2024

@author: ayhan
"""

import torch
import torch.nn as nn
import torch.optim as optim

class DQNPacman(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNPacman, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x