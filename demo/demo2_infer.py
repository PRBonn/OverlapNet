#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a demo to infer overlap and relative yaw angle between two LiDAR scans

import yaml
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/two_heads'))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from infer import *


def demo_infer(config, scan1, scan2):
  """ This function is used to infer overlap and yaw between two LiDAR scans.
  """
  # init the inferring class
  infer = Infer(config)
  
  # infer overlap and yaw between two scans
  overlap, yaw = infer.infer_one(scan1, scan2)
  
  # get the corresponding range data
  preprocess_data_folder = os.path.join(infer.datasetpath, infer.seq, 'depth')
  depth_data = []
  for filename in infer.filenames:
    depth_path = os.path.join(preprocess_data_folder, filename + '.npy')
    depth_data.append(np.load(depth_path))

  # visualize the results
  plt.figure(figsize=(6, 2))
  gs1 = gridspec.GridSpec(2, 1)
  gs1.update(wspace=0.001, hspace=0.001)  # set the spacing between axes.
  
  ax0 = plt.subplot(gs1[0])
  ax0.set_title('scan1')
  ax0.imshow(depth_data[0])
  ax0.set_axis_off()
  
  ax1 = plt.subplot(gs1[1])
  ax1.set_title('scan2')
  ax1.imshow(depth_data[1])
  ax1.set_axis_off()
  
  plt.suptitle('Overlap: ' + str(overlap) + '  Yaw: ' + str(yaw))

  plt.show()
  
  
if __name__ == '__main__':
  # load configuration file
  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
    
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  network_config_filename = config['Demo2']['network_config']
  scan1_path = config['Demo2']['scan1_path']
  scan2_path = config['Demo2']['scan2_path']

  network_config = yaml.load(open(network_config_filename))
  network_config['infer_seqs'] = config['Demo2']['infer_seqs']
  
  # start demo2
  demo_infer(network_config, scan2_path, scan1_path)
