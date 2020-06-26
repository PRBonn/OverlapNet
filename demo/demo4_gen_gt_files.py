#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This demo shows how to generate the overlap and yaw ground truth files for training and testing.

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
from com_overlap_yaw import com_overlap_yaw
from normalize_data import normalize_data
from split_train_val import split_train_val
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


def vis_gt(xys, ground_truth_mapping):
  """Visualize the overlap value on trajectory"""
  # set up plot
  fig, ax = plt.subplots()
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
  mapper = cm.ScalarMappable(norm=norm)  # cmap="magma"
  mapper.set_array(ground_truth_mapping[:, 2])
  colors = np.array([mapper.to_rgba(a) for a in ground_truth_mapping[:, 2]])
  
  # sort according to overlap
  indices = np.argsort(ground_truth_mapping[:, 2])
  xys = xys[indices]
  
  ax.scatter(xys[:, 0], xys[:, 1], c=colors[indices], s=10)
  
  ax.axis('square')
  ax.set_xlabel('X [m]')
  ax.set_ylabel('Y [m]')
  ax.set_title('Demo 4: Generate ground truth for training')
  cbar = fig.colorbar(mapper, ax=ax)
  cbar.set_label('Overlap', rotation=270, weight='bold')
  plt.show()


if __name__ == '__main__':
  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  poses_file = config['Demo4']['poses_file']
  calib_file = config['Demo4']['calib_file']
  scan_folder = config['Demo4']['scan_folder']
  dst_folder = config['Demo4']['dst_folder']
  
  # load scan paths
  scan_paths = load_files(scan_folder)

  # load calibrations
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  # load poses
  poses = load_poses(poses_file)
  pose0_inv = np.linalg.inv(poses[0])

  # for KITTI dataset, we need to convert the provided poses 
  # from the camera coordinate system into the LiDAR coordinate system  
  poses_new = []
  for pose in poses:
    poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
  poses = np.array(poses_new)

  # generate overlap and yaw ground truth array
  ground_truth_mapping = com_overlap_yaw(scan_paths, poses, frame_idx=0)
  
  # normalize the distribution of ground truth data
  dist_norm_data = normalize_data(ground_truth_mapping)
  
  # split ground truth for training and validation
  train_data, validation_data = split_train_val(dist_norm_data)
  
  # add sequence label to the data and save them as npz files
  seq_idx = '07'
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'ground_truth')
  try:
    os.stat(dst_folder)
    print('generating depth data in: ', dst_folder)
  except:
    print('creating new depth folder: ', dst_folder)
    os.mkdir(dst_folder)
    
  # training data
  train_seq = np.empty((train_data.shape[0], 2), dtype=object)
  train_seq[:] = seq_idx
  np.savez_compressed(dst_folder + '/train_set', overlaps=train_data, seq=train_seq)
  
  # validation data
  validation_seq = np.empty((validation_data.shape[0], 2), dtype=object)
  validation_seq[:] = seq_idx
  np.savez_compressed(dst_folder + '/validation_set', overlaps=validation_data, seq=validation_seq)
  
  # raw ground truth data, fully mapping, could be used for testing
  ground_truth_seq = np.empty((ground_truth_mapping.shape[0], 2), dtype=object)
  ground_truth_seq[:] = seq_idx
  np.savez_compressed(dst_folder + '/ground_truth_overlap_yaw', overlaps=ground_truth_mapping, seq=ground_truth_seq)
  
  print('Finish saving the ground truth data for training and testing at: ', dst_folder)
  
  # visualize the raw ground truth mapping
  vis_gt(poses[:, :2, 3], ground_truth_mapping)
  
  





