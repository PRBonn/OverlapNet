import torch
import torch.nn as nn
import os
import numpy as np
import sys
import yaml
sys.path.append('../tools/')
sys.path.append('../modules/')
from fe_dl_dh import featureExtracter_deltaLayer_deltaHead
from feature_extracter import featureExtracter
from read_samples import read_one_need_from_seq_depth_normals_intensity
import time

class featureGenerator():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, batch_size=1, lr = 0.01, use_depth=True, use_intensity=True, use_normals=True,
                 data_root_folder=None, features_folder=None, pre_trained_weights=None ):
        super(featureGenerator, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.pre_trained_weights = pre_trained_weights
        self.data_root_folder = data_root_folder
        self.features_folder=features_folder
        self.use_depth = use_depth
        self.use_intensity = use_intensity
        self.use_normals = use_normals

        self.amodel = featureExtracter_deltaLayer_deltaHead(channels=self.channels)
        self.amodel_leg = featureExtracter(channels=self.channels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.amodel_leg.to(self.device)


    def eval(self):

        if os.path.exists(self.pre_trained_weights):
            checkpoint = torch.load(self.pre_trained_weights)
            self.amodel.load_state_dict(checkpoint['state_dict'])  # 加载状态字典
            print("loading pretrained weights....")
        else:
            print("Please check your pretained weights !")


        with torch.no_grad():
            for j in range(4541):
                print("generating the feature map of frame " + str(j) + " in " + features_folder)
                sample_batch_l = read_one_need_from_seq_depth_normals_intensity(str(j).zfill(6), "00", self.data_root_folder)
                extracted_l, extracted_r, overlap = self.amodel(sample_batch_l, sample_batch_l)
                
                extracted_l_numpy = extracted_l.detach().cpu().numpy()
                np.save(os.path.join(features_folder,str(j).zfill(6)), extracted_l_numpy)




if __name__ == '__main__':


    # load config file
    config_filename = '../config/config.yml'    
    config = yaml.load(open(config_filename),Loader=yaml.FullLoader)
    data_root_folder = config['dataHandler']["dataset_folder"]
    use_depth = config['dataHandler']["use_depth"]
    use_intensity = config['dataHandler']["use_intensity"]
    use_normals = config['dataHandler']["use_normals"]
    pretrained_model = config['testHandler']["pretrained_model"]
    features_folder = config['testHandler']["features_folder"]


    num_channels = 0
    if use_depth:
        num_channels += 1
    if use_intensity:
        num_channels += 1
    if use_normals:
        num_channels += 3   


    test_handler = featureGenerator(height=64, width=900, channels=num_channels, norm_layer=None, batch_size=1, lr=0.001,
                                use_depth=use_depth, use_intensity=use_intensity, use_normals=use_normals, data_root_folder = data_root_folder, 
                                features_folder=features_folder, pre_trained_weights=pretrained_model)

    test_handler.eval()