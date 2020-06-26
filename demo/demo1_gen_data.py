#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a demo to generate data and
#        visualize different cues generated from LiDAR scans as images

import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
import matplotlib.pyplot as plt
from utils import *
import gen_depth_data as gen_depth
import gen_normal_data as gen_normal
import gen_intensity_data as gen_intensity
import gen_semantic_data as gen_semantics


def show_images(depth_data, normal_data, intensity_data, semantic_data):
  """ This function is used to visualize different types of data
      generated from the LiDAR scan, including depth, normal, intensity and semantics.
  """
  fig, axs = plt.subplots(4, figsize=(6, 4))
  axs[0].set_title('range_data')
  axs[0].imshow(depth_data)
  axs[0].set_axis_off()
  
  axs[1].set_title('normal_data')
  axs[1].imshow(normal_data)
  axs[1].set_axis_off()
  
  axs[2].set_title('intensity_data')
  # truncate the intensity to better visualize
  intensity_data[intensity_data < 0] = 0
  axs[2].imshow(intensity_data, cmap='gray')
  axs[2].set_axis_off()
  
  axs[3].set_title('semantic_data')
  # color the semantic class to better visualize
  semantic_colors = [semantic_mapping[semantic_label][::-1]
                     for semantic_label
                     in np.argmax(semantic_data, axis=2).flatten()]

  axs[3].imshow(np.array(semantic_colors).reshape([semantic_data.shape[0], semantic_data.shape[1], 3]))
  axs[3].set_axis_off()

  plt.suptitle('Preprocessed data from the LiDAR scan')
  plt.show()


def gen_data(semantic_folder, scan_folder, dst_folder, visualize=True):
  """ This function is used to generate different types of data
      from the LiDAR scan, including depth, normal, intensity and semantics.
  """
  range_data = gen_depth.gen_depth_data(scan_folder, dst_folder)[0]
  normal_data = gen_normal.gen_normal_data(scan_folder, dst_folder)[0]
  intensity_data = gen_intensity.gen_intensity_data(scan_folder, dst_folder)[0]
  semantic_data = gen_semantics.gen_semantic_data(semantic_folder, scan_folder, dst_folder)[0]

  if visualize:
    show_images(range_data, normal_data, intensity_data, semantic_data)


if __name__ == '__main__':
  # load config file
  config_filename = 'config/demo.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  scan_folder = config['Demo1']["scan_folder"]
  semantic_folder = config['Demo1']["semantic_folder"]
  dst_folder = config['Demo1']["dst_folder"]
  
  # start the demo1 to generate different types of data from LiDAR scan
  gen_data(semantic_folder, scan_folder, dst_folder)
