import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../tools/')
sys.path.append('../modules/')
from feature_extracter import featureExtracter
from delta_layer import deltaLayer
from delta_head import deltaHead
from read_samples import read_one_need_from_seq

class deltaLayer_deltaHead(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None ):
        super(deltaLayer_deltaHead, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.deltaLayer = deltaLayer(negateDiffs=False)
        self.deltaHead = deltaHead()

    def forward(self, x_l, x_r):
        diff = self.deltaLayer(x_l, x_r)
        overlap = self.deltaHead(diff)

        return overlap


