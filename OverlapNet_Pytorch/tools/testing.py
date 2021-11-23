import torch
import torch.nn as nn
import os
import numpy as np
import sys
from tensorboardX import SummaryWriter
sys.path.append('../tools/')
sys.path.append('../modules/')
from tools.gt_unzip import overlap_orientation_npz_file2string_string_nparray
from modules.fe_dl_dh import featureExtracter_deltaLayer_deltaHead
from tools.read_samples import read_one_batch_test
from tools.read_samples import read_one_batch
from tools.read_samples import read_one_need
np.set_printoptions(threshold=sys.maxsize)

class testHandler():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, batch_size=6, lr = 0.01,
                 gt_overlap_yaw=None,train_set=None,validation_set=None,use_transformer=True):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.gt_overlap_yaw = gt_overlap_yaw
        self.train_set = train_set
        self.validation_set = validation_set

        self.amodel = featureExtracter_deltaLayer_deltaHead(channels=5, use_transformer=self.use_transformer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        # self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)


    def eval(self):

        epochs = 1
        pre_trained_weights = '/home/mjy/dev/aOverlapNet/amodel.pt'
        if os.path.exists(pre_trained_weights):
            checkpoint = torch.load(pre_trained_weights)
            self.amodel.load_state_dict(checkpoint)
            print("loading pretrained weights....")

        sample_list = os.listdir("/home/mjy/dev/aOverlapNet/data/depth")
        current_batch = read_one_need("000000")
        current_batch = current_batch.repeat(self.batch_size, 1, 1, 1).type(torch.FloatTensor).cuda()

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        old_image = current_batch
        with torch.no_grad():
            for i in range(epochs):
                # one batch
                # print(len(self.train_set[0]) // self.batch_size)
                loss_each_epoch = 0
                diff_each_epoch = 0
                # used_num = 0
                for j in range(len(self.train_set[0])//self.batch_size):
                    # (batch_size, 5, 64, 900)
                    # start --> end
                    # judge whether exceed .... TODO:
                    if (j+1)*self.batch_size>len(self.train_set[0]):
                        break
                    # reference --- sample_batch
                    sample_batch,sample_truth = read_one_batch(j*self.batch_size, (j+1)*self.batch_size,
                                                               self.gt_overlap_yaw,self.train_set,self.validation_set)
                    # print(sample_truth)
                    # current --- current_batch
                    self.amodel.eval()
                    extracted_l, extracted_r, overlap = self.amodel(current_batch, sample_batch)

                    diff_value = torch.abs(overlap[0,0,0] - sample_truth[0,0])
                    sigmoidx = (diff_value + 0.25) * 24 - 12
                    loss = torch.mean(1 / (1 + torch.exp(-sigmoidx)))
                    loss_each_epoch = loss.item()
                    # diff_each_epoch = diff_value.item()

                    # print("loss_each_epoch: ", loss_each_epoch)
                    # print("diff_each_epoch: ", diff_each_epoch)


                    overlap = overlap.item()
                    if sample_truth[0,0].item()>0.3:
                        print("truth:", sample_truth[0,0].item())
                        print("overlap:", overlap)
                        print("loss_each_epoch: ", loss_each_epoch)
                    if sample_truth[0,0].item()>0.3 and overlap > 0.3:
                        true_positive = true_positive + 1
                    elif sample_truth[0,0].item()<0.3 and overlap < 0.3:
                        true_negative  = true_negative + 1
                    elif sample_truth[0,0].item() < 0.3 and overlap > 0.3:
                        false_positive = false_positive + 1
                    elif sample_truth[0,0].item() > 0.3 and overlap < 0.3:
                        false_negative = false_negative + 1
                    # used_num = used_num + 1
                    recall = true_positive / (true_positive + false_negative+1e-4)
                    precison = true_positive / (true_positive + false_positive+1e-4)
                    # print("true_positive {} true_negative {} false_positive {} false_negative {}".format(
                    #     true_positive, true_negative, false_positive, false_negative))
                    # print("recall {} precision {}".format(recall, precison))




if __name__ == '__main__':

    # data
    gt_overlap_yaw = overlap_orientation_npz_file2string_string_nparray(["/home/mjy/dev/aOverlapNet/data/ground_truth/ground_truth_overlap_yaw.npz"])
    # train_set.npz
    train_set = overlap_orientation_npz_file2string_string_nparray(["/home/mjy/dev/aOverlapNet/data/ground_truth/train_set.npz"])
    # validation_set.npz
    validation_set = overlap_orientation_npz_file2string_string_nparray(["/home/mjy/dev/aOverlapNet/data/ground_truth/validation_set.npz"])

    test_handler = testHandler(height=64, width=900, channels=5, norm_layer=None, batch_size=1, lr=0.001,
                                 gt_overlap_yaw=gt_overlap_yaw, train_set=train_set, validation_set=validation_set,
                                     use_transformer=False)

    test_handler.eval()