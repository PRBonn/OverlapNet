import sys
sys.path.append('../tools/')
sys.path.append('../modules/')
import numpy as np
import torch
import copy
from gt_unzip import overlap_orientation_npz_file2string_string_nparray
import cv2
import os
import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt



def show_images_overlap_depth_64900(depth_data, depth_data2, overlap):
  """ This function is used to visualize different types of data
      generated from the LiDAR scan, including depth, normal, intensity and semantics.
  """

  depth_data = depth_data.detach().cpu().numpy()
  depth_data2 = depth_data2.detach().cpu().numpy()


  fig, axs = plt.subplots(2, figsize=(12, 10))
  axs[0].set_title('range_data1')
  axs[0].imshow(depth_data.reshape(64,900,1))

  axs[0].set_axis_off()

  axs[1].set_title('range_data2')
  axs[1].imshow(depth_data2.reshape(64,900,1))
  axs[1].set_axis_off()

  plt.suptitle('Preprocessed data from the LiDAR scan' + " overlap: " + str(overlap))
  plt.show()




def read_one_need_from_seq(file_num, seq_num, data_root_folder):


    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))

    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def read_one_need_from_seq_depth_normals(file_num, seq_num, data_root_folder):


    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))
    normal_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/normal_map/" + file_num + ".png"),dtype=np.float32)
    normal_data_cp = copy.deepcopy(normal_data)
    normal_data = copy.deepcopy(normal_data_cp)
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    normal_data_tensor = torch.from_numpy(normal_data).type(torch.FloatTensor).cuda()
    normal_data_tensor = normal_data_tensor.permute(2, 0, 1)
    normal_data_tensor = torch.unsqueeze(normal_data_tensor, dim=0)

    combined_tensor = torch.cat((depth_data_tensor, normal_data_tensor), dim=1)
    return combined_tensor




def read_one_need_from_seq_depth_intensity(file_num, seq_num, data_root_folder):

    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))

    intensity_data = \
        np.load(data_root_folder + seq_num + "/intensity_map/" + file_num + ".npz")['arr_0']

    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    intensity_data_tensor = torch.from_numpy(intensity_data).type(torch.FloatTensor).cuda()
    intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)
    intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)

    combined_tensor = torch.cat((depth_data_tensor, intensity_data_tensor), dim=1)

    return combined_tensor






def read_one_need_from_seq_depth_normals_intensity(file_num, seq_num, data_root_folder):

    depth_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
                            cv2.IMREAD_GRAYSCALE))
    normal_data = \
        np.array(cv2.imread(data_root_folder + seq_num + "/normal_map/" + file_num + ".png"),dtype=np.float32)

    intensity_data = \
        np.load(data_root_folder + seq_num + "/intensity_map/" + file_num + ".npz")['arr_0']

    normal_data_cp = copy.deepcopy(normal_data)
    normal_data = copy.deepcopy(normal_data_cp)
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    normal_data_tensor = torch.from_numpy(normal_data).type(torch.FloatTensor).cuda()
    normal_data_tensor = normal_data_tensor.permute(2, 0, 1)
    normal_data_tensor = torch.unsqueeze(normal_data_tensor, dim=0)
    intensity_data_tensor = torch.from_numpy(intensity_data).type(torch.FloatTensor).cuda()
    intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)
    intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)

    combined_tensor = torch.cat((depth_data_tensor, intensity_data_tensor), dim=1)
    combined_tensor = torch.cat((combined_tensor, normal_data_tensor), dim=1)

    return combined_tensor




# def read_one_need_from_seq_depth_intensity_semantic(file_num, seq_num):

#     data_root_folder = "/home/mjy/datasets/overlapnet_datasets/OverlapNet/kitti/dataset_full/"

#     depth_data = \
#         np.array(cv2.imread(data_root_folder + seq_num + "/depth_map/" + file_num + ".png",
#                             cv2.IMREAD_GRAYSCALE))

#     intensity_data = \
#         np.load(data_root_folder + seq_num + "/intensity_map/" + file_num + ".npz")['arr_0']

#     semantic_data = \
#         np.load(data_root_folder + seq_num + "/probability_map_npz_pca/" + file_num + ".npz")['arr_0']

#     depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
#     depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
#     depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

#     intensity_data_tensor = torch.from_numpy(intensity_data).type(torch.FloatTensor).cuda()
#     intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)
#     intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)

#     semantic_data_tensor = torch.from_numpy(semantic_data).type(torch.FloatTensor).cuda()
#     semantic_data_tensor = semantic_data_tensor.permute(2, 0, 1)
#     semantic_data_tensor = torch.unsqueeze(semantic_data_tensor, dim=0)

#     combined_tensor = torch.cat((depth_data_tensor, intensity_data_tensor), dim=1)
#     combined_tensor = torch.cat((combined_tensor, semantic_data_tensor), dim=1)

#     return combined_tensor





def read_one_batch(start, end, data_root_folder, train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, use_depth=True, use_intensity=True, use_normals=True):
    num_channels = 0
    if use_depth:
        num_channels += 1
    if use_intensity:
        num_channels += 1
    if use_normals:
        num_channels += 3   

    batch_size = end - start
    sample_batch_l = torch.from_numpy(np.zeros((batch_size, num_channels, 64, 900))).type(torch.FloatTensor).cuda()
    sample_batch_r = torch.from_numpy(np.zeros((batch_size, num_channels, 64, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()
    # data_root_folder = "/home/mjy/datasets/overlapnet_datasets/OverlapNet/kitti/dataset_full/"
    for i in range(batch_size):
        if use_depth:
            depth_data = \
                np.array(cv2.imread(data_root_folder + train_dir1[start+i] + "/depth_map/"+ train_imgf1[start+i] + ".png",
                            cv2.IMREAD_GRAYSCALE))
            depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
            # depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

            depth_data2 = \
            np.array(cv2.imread(
                data_root_folder + train_dir2[start + i] + "/depth_map/" + train_imgf2[start + i] + ".png",
                cv2.IMREAD_GRAYSCALE))
            depth_data_tensor2 = torch.from_numpy(depth_data2).type(torch.FloatTensor).cuda()
            # depth_data_tensor2 = torch.unsqueeze(depth_data_tensor2, dim=0)

            # print(data_root_folder + train_dir1[start+i] + "/depth_map/"+ train_imgf1[start+i] + ".png")
            # print(data_root_folder + train_dir2[start + i] + "/depth_map/" + train_imgf2[start + i] + ".png")
            

        if use_intensity:
            intensity_data = \
                np.load(data_root_folder + train_dir1[start+i] + "/intensity_map/"+ train_imgf1[start+i] + ".npz")['arr_0']
            intensity_data_tensor = torch.from_numpy(intensity_data).type(torch.FloatTensor).cuda()
            # intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)

            intensity_data2 = \
                np.load(data_root_folder + train_dir2[start + i] + "/intensity_map/" + train_imgf2[start + i] + ".npz")['arr_0']
            intensity_data_tensor2 = torch.from_numpy(intensity_data2).type(torch.FloatTensor).cuda()
            # intensity_data_tensor2 = torch.unsqueeze(intensity_data_tensor2, dim=0)

        if use_normals:
            normal_data = \
                np.array(cv2.imread(data_root_folder + train_dir1[start+i] + "/normal_map/"+ train_imgf1[start+i] + ".png"))
            normal_data_tensor = torch.from_numpy(normal_data).type(torch.FloatTensor).cuda()
            normal_data_tensor = normal_data_tensor.permute(2, 0, 1)

            normal_data2 = \
                np.array(cv2.imread(
                    data_root_folder + train_dir2[start + i] + "/normal_map/" + train_imgf2[start + i] + ".png"))
            normal_data_tensor2 = torch.from_numpy(normal_data2).type(torch.FloatTensor).cuda()
            normal_data_tensor2 = normal_data_tensor2.permute(2, 0, 1)



        if num_channels==5:
            depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
            intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)
            combined_tensor = torch.cat((depth_data_tensor, intensity_data_tensor), dim=0)
            combined_tensor = torch.cat((combined_tensor, normal_data_tensor), dim=0)
            sample_batch_l[i,:,:,:] = combined_tensor

            depth_data_tensor2 = torch.unsqueeze(depth_data_tensor2, dim=0)
            intensity_data_tensor2 = torch.unsqueeze(intensity_data_tensor2, dim=0)
            combined_tensor2 = torch.cat((depth_data_tensor2, intensity_data_tensor2), dim=0)
            combined_tensor2 = torch.cat((combined_tensor2, normal_data_tensor2), dim=0)
            sample_batch_r[i,:,:,:] = combined_tensor2

        elif num_channels == 4:
            sample_batch_l[i,1:4,:,:] = normal_data_tensor
            if use_depth:
                sample_batch_l[i,0,:,:] = depth_data_tensor
            elif use_intensity:
                sample_batch_l[i,0,:,:] = intensity_data_tensor

            sample_batch_r[i,1:4,:,:] = normal_data_tensor2
            if use_depth:
                sample_batch_r[i,0,:,:] = depth_data_tensor2
            elif use_intensity:
                sample_batch_r[i,0,:,:] = intensity_data_tensor2

        elif num_channels == 3:
            sample_batch_l[i,:,:,:] = normal_data_tensor

            sample_batch_r[i,:,:,:] = normal_data_tensor2

        elif num_channels == 2:
            depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
            intensity_data_tensor = torch.unsqueeze(intensity_data_tensor, dim=0)
            combined_tensor = torch.cat((depth_data_tensor, intensity_data_tensor), dim=0)
            sample_batch_l[i,:,:,:] = combined_tensor

            depth_data_tensor2 = torch.unsqueeze(depth_data_tensor2, dim=0)
            intensity_data_tensor2 = torch.unsqueeze(intensity_data_tensor2, dim=0)
            combined_tensor2 = torch.cat((depth_data_tensor2, intensity_data_tensor2), dim=0)
            sample_batch_r[i,:,:,:] = combined_tensor2

        elif num_channels == 1:
            if use_depth:
                sample_batch_l[i,0,:,:] = depth_data_tensor
            elif use_intensity:
                sample_batch_l[i,0,:,:] = intensity_data_tensor

            if use_depth:
                sample_batch_r[i,0,:,:] = depth_data_tensor2
            elif use_intensity:
                sample_batch_r[i,0,:,:] = intensity_data_tensor2


        sample_truth[i, :] = torch.from_numpy(np.array(train_overlap[start+i])).type(torch.FloatTensor).cuda()

    return sample_batch_l, sample_batch_r, sample_truth


if __name__ == '__main__':

    data_root_folder = "/home/mjy/datasets/overlapnet_datasets/OverlapNet/kitti/dataset_full/"
    training_seqs = ["07","08"]

    traindata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/train_set.npz') for seq in training_seqs]
    validationdata_npzfiles = [os.path.join(data_root_folder, seq, 'overlaps/test_set.npz') for seq in training_seqs]

    (train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap) = \
        overlap_orientation_npz_file2string_string_nparray(traindata_npzfiles, shuffle=False)


    f1_index = "003755"
    f1_seq = "08"

    this_frame1 = read_one_need_from_seq_depth_intensity("000887", "08", data_root_folder)
    this_frame2 = read_one_need_from_seq_depth_intensity("000227", "08", data_root_folder)

    show_images_overlap_depth_64900(this_frame1[0, 1, :, :], this_frame2[0, 1, :, :], 0)


