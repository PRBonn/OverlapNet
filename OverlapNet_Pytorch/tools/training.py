import os
import numpy as np
import torch
import torch.nn as nn
import yaml
from tensorboardX import SummaryWriter
import sys
sys.path.append('../tools/')
sys.path.append('../modules/')
from fe_dl_dh import featureExtracter_deltaLayer_deltaHead
from read_samples import read_one_batch
from read_samples import read_one_need_from_seq
from read_samples import read_one_need_from_seq_depth_intensity
from read_samples import read_one_need_from_seq_depth_normals
from read_samples import read_one_need_from_seq_depth_normals_intensity

from gt_unzip import overlap_orientation_npz_file2string_string_nparray
from tqdm import tqdm

class trainHandler():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, batch_size=6, lr = 0.001, use_depth=True, use_intensity=True, use_normals=True,
                 data_root_folder=None, train_set=None,validation_set=None):
        super(trainHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.use_depth = use_depth
        self.use_intensity = use_intensity
        self.use_normals = use_normals
        self.train_set = train_set
        self.validation_set = validation_set
        self.data_root_folder = data_root_folder

        self.amodel = featureExtracter_deltaLayer_deltaHead(channels=channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        # self.parameters = filter(lambda p: p.requires_grad,
        #                          self.amodel.parameters())
        self.parameters  = self.amodel.parameters()
        # self.optimizer = torch.optim.Adam(self.parameters, self.learning_rate)
        self.optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate, momentum=0.9)

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.65)

        self.traindata_npzfiles = train_set
        self.validationdata_npzfiles = validation_set

        (self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap) = \
            overlap_orientation_npz_file2string_string_nparray(self.traindata_npzfiles)

        (self.validation_imgf1, self.validation_imgf2, self.validation_dir1, self.validation_dir2, self.validation_overlap) = \
            overlap_orientation_npz_file2string_string_nparray(self.validationdata_npzfiles, shuffle=False)

        self.resume = False
        self.save_name = "amodel.pth.tar"

    def train(self):

        # print(self.amodel)

        epochs = 100

        if self.resume:  
            resume_filename = self.save_name
            print("Resuming From ", resume_filename)
            checkpoint = torch.load(resume_filename)
            starting_epoch = checkpoint['epoch']
            self.amodel.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Training From Scratch ..." )
            starting_epoch = 0

        writer1 = SummaryWriter(comment="LR_xxx_Seq_xxx")
        for i in range(starting_epoch, epochs):

            loss_each_epoch = 0
            used_num = 0
            loss_max= 0

            # for j in tqdm(range(len(self.train_imgf1)//self.batch_size)):
            for j in range(len(self.train_imgf1) // self.batch_size):
                if (j+1)*self.batch_size>len(self.train_imgf1):
                    break

                sample_batch_l, sample_batch_r, sample_truth = read_one_batch(j*self.batch_size, (j+1)*self.batch_size,  self.data_root_folder, 
                                                           self.train_imgf1, self.train_imgf2, self.train_dir1, self.train_dir2, self.train_overlap,
                                                           use_depth=self.use_depth, use_intensity=self.use_intensity, use_normals=self.use_normals)
                sample_batch_l.requires_grad_(True)
                sample_batch_r.requires_grad_(True)
                self.amodel.train()
                self.optimizer.zero_grad() 
                extracted_l, extracted_r, overlap = self.amodel(sample_batch_l, sample_batch_r)

                diff_value = torch.abs(overlap.squeeze(-1) - sample_truth)
                sigmoidx = (diff_value[:,0] + 0.25) * 24 - 12
                loss = torch.mean(1 / (1 + torch.exp(-sigmoidx)))

                loss.backward()
                self.optimizer.step()
                loss_each_epoch = loss_each_epoch + loss.item()
                used_num = used_num + 1

                print("epoch " + str(i) + " ---> " + "batch " + str(j) + " loss: ", loss)


            print("epoch {} loss {}".format(i, loss_each_epoch/used_num))
            print("saving weights ...")
            self.scheduler.step()

            torch.save({
                'epoch': i,
                'state_dict': self.amodel.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
                self.save_name)

            print("Model Saved As " + 'amodel.pth.tar')

            writer1.add_scalar("loss", loss_each_epoch / used_num, global_step=i)




if __name__ == '__main__':

    # load config file
    config_filename = '../config/config.yml'    
    config = yaml.load(open(config_filename),Loader=yaml.FullLoader)
    data_root_folder = config['dataHandler']["dataset_folder"]
    use_depth = config['dataHandler']["use_depth"]
    use_intensity = config['dataHandler']["use_intensity"]
    use_normals = config['dataHandler']["use_normals"]
    train_seqs = config['trainHandler']["train_seqs"]
    valid_seqs = config['trainHandler']["valid_seqs"]


    print("train_seqs: ", train_seqs)
    print("valid_seqs: ", valid_seqs)


    traindata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/train_set.npz') for seq in train_seqs]
    validationdata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/test_set.npz') for seq in valid_seqs]

    num_channels = 0
    if use_depth:
        num_channels += 1
    if use_intensity:
        num_channels += 1
    if use_normals:
        num_channels += 3   

    train_handler = trainHandler(height=64, width=900, channels=num_channels, norm_layer=None, batch_size=2, lr=0.01,
                                use_depth=use_depth, use_intensity=use_intensity, use_normals=use_normals,data_root_folder = data_root_folder,
                                 train_set=traindata_npzfiles, validation_set=validationdata_npzfiles )

    train_handler.train()