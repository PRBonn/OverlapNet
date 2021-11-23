import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../tools/')
# from read_samples import read_one_sample


class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d   # number of channels


        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,15), stride=2, bias=True)
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,15), stride=(2,1), bias=True)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,15), stride=(2,1), bias=True)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,12), stride=(2,1), bias=True)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,9), stride=(2,1), bias=True)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,9), stride=(2,1), bias=True)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,9), stride=(2,1), bias=True)
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,9), stride=(2,1), bias=True)
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,7), stride=(2,1), bias=True)
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,5), stride=(2,1), bias=True)
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,3), stride=(2,1), bias=True)
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        self.convLast = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1), bias=True)

    def forward(self, x_l, x_r):

        # out_l = self.relu(self.bn1(self.conv1(x_l)))
        # out_l = self.relu(self.bn2(self.conv2(out_l)))
        # out_l = self.relu(self.bn3(self.conv3(out_l)))
        # out_l = self.relu(self.bn4(self.conv4(out_l)))
        # out_l = self.relu(self.bn5(self.conv5(out_l)))
        # out_l = self.relu(self.bn6(self.conv6(out_l)))
        # out_l = self.relu(self.bn7(self.conv7(out_l)))
        # out_l = self.relu(self.bn8(self.conv8(out_l)))
        # out_l = self.relu(self.bn9(self.conv9(out_l)))
        # out_l = self.relu(self.bn10(self.conv10(out_l)))
        # out_l = self.relu(self.bn11(self.conv11(out_l)))
        #
        # out_r = self.relu(self.bn1(self.conv1(x_r)))
        # out_r = self.relu(self.bn2(self.conv2(out_r)))
        # out_r = self.relu(self.bn3(self.conv3(out_r)))
        # out_r = self.relu(self.bn4(self.conv4(out_r)))
        # out_r = self.relu(self.bn5(self.conv5(out_r)))
        # out_r = self.relu(self.bn6(self.conv6(out_r)))
        # out_r = self.relu(self.bn7(self.conv7(out_r)))
        # out_r = self.relu(self.bn8(self.conv8(out_r)))
        # out_r = self.relu(self.bn9(self.conv9(out_r)))
        # out_r = self.relu(self.bn10(self.conv10(out_r)))
        # out_r = self.relu(self.bn11(self.conv11(out_r)))

        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        out_l = self.relu(self.conv10(out_l))
        out_l = self.relu(self.conv11(out_l))

        out_r = self.relu(self.conv1(x_r))
        out_r = self.relu(self.conv2(out_r))
        out_r = self.relu(self.conv3(out_r))
        out_r = self.relu(self.conv4(out_r))
        out_r = self.relu(self.conv5(out_r))
        out_r = self.relu(self.conv6(out_r))
        out_r = self.relu(self.conv7(out_r))
        out_r = self.relu(self.conv8(out_r))
        out_r = self.relu(self.conv9(out_r))
        out_r = self.relu(self.conv10(out_r))
        out_r = self.relu(self.conv11(out_r))


        return out_l, out_r


if __name__ == '__main__':
    combined_tensor = read_one_sample()

    feature_extracter=featureExtracter(use_transformer=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()
    output_l, output_r = feature_extracter(combined_tensor, combined_tensor)
    print(output_l)
    print(output_l.shape)   # torch.Size([1, 128, 1, 360])
    # print(output_r)