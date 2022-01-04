import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../tools/')


class deltaHead(nn.Module):
    def __init__(self, norm_layer=None):
        super(deltaHead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # number of channels
        self.conv1 = nn.Conv2d(128, 64, kernel_size=(1,15), stride=(1,15), bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(15,1), stride=(15,1), bias=False)
        self.bn2 = norm_layer(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), bias=False)
        self.bn3 = norm_layer(256)
        self.linear1 = nn.Linear(22*22*256, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x= x.view(x.shape[0], 1, -1)
        x = self.sigmoid(self.linear1(x))

        return x
