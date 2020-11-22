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


def demo_infer(config, files, target_id, ref_id):
  """ This function is used to infer overlap and yaw between multiple LiDAR scans.
  """
  # init the inferring class
  infer = Infer(config)
  
  # infer overlap and yaw between multiple scans
  overlap, yaw = infer.infer_multiple(files, target_id, ref_id)
  print('overlap:',overlap)
  print('yaw:',yaw)
  
  
if __name__ == '__main__':
  # load configuration file
  config_filename = os.path.abspath('./config/multiple.yml')
  
  if len(sys.argv) > 1:
    config_filename = os.path.abspath(sys.argv[1])
    
  config = yaml.load(open(config_filename))
  
  # start demo2
  demo_infer(config,["000000","000001"], [0, 0, 1, 1],[ 0, 1, 0, 1])
