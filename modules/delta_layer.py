import torch
import torch.nn as nn
import numpy as np
import random

# # input : # out_r (bs, 128, 1, 360)
# def deltaLayer(encoded_l, encoded_r, negateDiffs=False):

#     bs = encoded_l.shape[0]
#     w = encoded_l.shape[-1]
#     h = encoded_l.shape[-2]
#     chan = encoded_l.shape[-3]  # (bs, 128, 1, 360)
#     reshaped_l = encoded_l.permute(0,1,2,3)    # (bs, 128, 1, 360)
#     reshaped_r = encoded_r.permute(0,1,3,2)  # (bs, 128, 360, 1)


#     tiled_l = reshaped_l.repeat(1, 1, w * h, 1)   # (bs, 128, 360, 360)
#     tiled_r = reshaped_r.repeat(1, 1, 1, w * h)   # (bs, 128, 360, 360)
#     diff = torch.abs(tiled_l - tiled_r)
#     # if negateDiffs:
#     #     diff = Lambda(lambda x: -K.abs(x[0] - x[1]))([tiled_l, tiled_r])
#     # else:
#     #     diff = Lambda(lambda x: K.abs(x[0] - x[1]))([tiled_l, tiled_r])
#     return diff


class deltaLayer(nn.Module):
    def __init__(self, negateDiffs=False):
        super(deltaLayer, self).__init__()
        self.negateDiffs = negateDiffs


    def forward(self, encoded_l, encoded_r):

        bs = encoded_l.shape[0]
        w = encoded_l.shape[-1]
        h = encoded_l.shape[-2]
        chan = encoded_l.shape[-3]  # (bs, 128, 1, 360)
        reshaped_l = encoded_l.permute(0,1,2,3)    # (bs, 128, 1, 360)
        reshaped_r = encoded_r.permute(0,1,3,2)  # (bs, 128, 360, 1)


        tiled_l = reshaped_l.repeat(1, 1, w * h, 1)   # (bs, 128, 360, 360)
        tiled_r = reshaped_r.repeat(1, 1, 1, w * h)   # (bs, 128, 360, 360)
        diff = torch.abs(tiled_l - tiled_r)

        # TODO:
        # if self.negateDiffs:
        #     diff = Lambda(lambda x: -K.abs(x[0] - x[1]))([tiled_l, tiled_r])
        # else:
        #     diff = Lambda(lambda x: K.abs(x[0] - x[1]))([tiled_l, tiled_r])

        return diff
