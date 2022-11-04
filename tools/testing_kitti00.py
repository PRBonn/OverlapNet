import torch
import torch.nn as nn
import os
import numpy as np
import sys
import yaml
sys.path.append('../tools/')
sys.path.append('../modules/')
from fe_dl_dh import featureExtracter_deltaLayer_deltaHead
from dl_dh import deltaLayer_deltaHead
from feature_extracter import featureExtracter
from tqdm import tqdm
import time

class testHandler():
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, batch_size=1, lr = 0.01, use_depth=True, use_intensity=True, use_normals=True,
                 data_root_folder=None, features_folder=None, ground_truth_file_name=None, pre_trained_weights=None ):
        super(testHandler, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.norm_layer = norm_layer
        self.batch_size = batch_size
        self.learning_rate = lr
        self.ground_truth_file_name = ground_truth_file_name
        self.pre_trained_weights = pre_trained_weights
        self.data_root_folder = data_root_folder
        self.features_folder = features_folder
        self.use_depth = use_depth
        self.use_intensity = use_intensity
        self.use_normals = use_normals

        self.amodel = deltaLayer_deltaHead(channels=self.channels)
        self.amodel_leg = featureExtracter(channels=self.channels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amodel.to(self.device)
        self.amodel_leg.to(self.device)


    def eval(self):


        ground_truth = np.load(self.ground_truth_file_name, allow_pickle='True')['arr_0']
        pos_num = 0
        for idx in range(len(ground_truth) - 1):
            gt_idxes = ground_truth[int(idx)]
            if gt_idxes.any():
                pos_num = pos_num + 1


        epochs = 1
        if os.path.exists(self.pre_trained_weights):
            model_dict = self.amodel.state_dict()
            pretrained_dict = torch.load(self.pre_trained_weights)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.amodel.load_state_dict(model_dict)
            print("loading pretrained weights....")
        else:
            print("Please check your pretained weights !")


        with torch.no_grad():
            all_time = 0
            all_fe_time = 0
            use_num = 0

            recall_list = []
            precision_list = []

            feature_list = os.listdir(self.features_folder)

            for thresh in np.arange(0.3, 0.32, 0.005):

                true_positive = 0
                true_negative = 0
                false_positive = 0
                false_negative = 0


                for j in range(0,4541):
                    print(str(thresh)+"----->",j)

                    sample_batch_l = np.load( os.path.join(self.features_folder, feature_list[j]))
                    sample_batch_l = torch.from_numpy(sample_batch_l).type(torch.FloatTensor).cuda()
                    max_overlap = 0
                    max_idx = -1
                    gt_idxes = ground_truth[int(j)]

                    
                    all_delta_time = 0
                    for k in range(0, j-100):

                        sample_batch_r = np.load(os.path.join(self.features_folder, feature_list[k]))
                        self.amodel.eval()
                        sample_batch_r = torch.from_numpy(sample_batch_r).type(torch.FloatTensor).cuda()
                        # time1 = time.time()
                        overlap = self.amodel(sample_batch_l, sample_batch_r)
                        overlap = overlap.item()
                        if max_overlap < overlap:
                            max_overlap = overlap
                            max_idx = int(k)
                        # time2 = time.time()
                    #     all_delta_time=all_delta_time+time2-time1
                    # print(all_delta_time)

                    
                    if max_idx in gt_idxes and max_overlap > thresh:
                        true_positive = true_positive + 1
                    elif (not (max_idx in gt_idxes)) and max_overlap < thresh:
                        true_negative  = true_negative + 1
                    elif (not (max_idx in gt_idxes)) and max_overlap > thresh:
                        false_positive = false_positive + 1
                    elif max_idx in gt_idxes  and max_overlap < thresh:
                        false_negative = false_negative + 1
                    # used_num = used_num + 1
                recall = true_positive / (true_positive + false_negative+1e-4)
                precison = true_positive / (true_positive + false_positive+1e-4)
                # print("true_positive {} true_negative {} false_positive {} false_negative {}".format(
                #     true_positive, true_negative, false_positive, false_negative))
                print("recall {} precision {}".format(recall, precison))
                recall_list.append(recall)
                precision_list.append(precison)
                
                np.save("./recall_results", np.array(recall_list))
                np.save("./precision_results", np.array(precision_list))




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
    ground_truth_file_name = config['testHandler']["gt_path"]


    num_channels = 0
    if use_depth:
        num_channels += 1
    if use_intensity:
        num_channels += 1
    if use_normals:
        num_channels += 3   


    test_handler = testHandler(height=64, width=900, channels=num_channels, norm_layer=None, batch_size=1, lr=0.001,
                                use_depth=use_depth, use_intensity=use_intensity, use_normals=use_normals, 
                                data_root_folder = data_root_folder, features_folder=features_folder,
                                 ground_truth_file_name=ground_truth_file_name, pre_trained_weights=pretrained_model)

    test_handler.eval()
