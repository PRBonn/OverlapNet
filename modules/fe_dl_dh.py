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

class featureExtracter_deltaLayer_deltaHead(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None ):
        super(featureExtracter_deltaLayer_deltaHead, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.featureExtracter = featureExtracter(height=self.height, width=self.width, channels=self.channels)
        self.deltaLayer = deltaLayer(negateDiffs=False)
        self.deltaHead = deltaHead()

    def forward(self, x_l, x_r):
        extracted_l, extracted_r = self.featureExtracter(x_l, x_r)
        diff = self.deltaLayer(extracted_l, extracted_r)
        overlap = self.deltaHead(diff)

        return extracted_l, extracted_r, overlap

if __name__ == '__main__':

    data_root_folder = "your_path_to_dataset/dataset_full/"

    combined_tensor1 = read_one_need_from_seq("000000", "00", data_root_folder)
    combined_tensor2 = read_one_need_from_seq("000000", "00", data_root_folder)
    featureExtracter_deltaLayer_deltaHead_Net = featureExtracter_deltaLayer_deltaHead(channels=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featureExtracter_deltaLayer_deltaHead_Net.to(device)
    featureExtracter_deltaLayer_deltaHead_Net.eval()
    extracted_l, extracted_r, diff, overlap = featureExtracter_deltaLayer_deltaHead_Net(combined_tensor1, combined_tensor2)

    print(overlap)

