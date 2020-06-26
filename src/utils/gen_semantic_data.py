#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a script to generate semantic data
import os
import numpy as np
try: from utils import load_files, range_projection
except: from utils import load_files, range_projection


def gen_semantic_data(semantic_folder, scan_folder, dst_folder, proj_H=64, proj_W=900):
  """ Generate projected semantic data in the shape of (64, 900, 20).
      The input raw data are in the shape of (Num_points, 20).
  """
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'semantic')
  try:
    os.stat(dst_folder)
    print('generating semantic data in: ', dst_folder)
  except:
    print('creating new semantic folder: ', dst_folder)
    os.mkdir(dst_folder)
    
  # load raw semantic predictions
  prob_paths = load_files(semantic_folder)
  
  # load corresponding LiDAR scans
  scan_paths = load_files(scan_folder)
  semantics = []
  # iterate over all semantic files in the given folder
  for idx in range(len(prob_paths)):
    # read semantic probabilities from the raw file
    probs = np.fromfile(prob_paths[idx], dtype=np.float32).reshape((-1, 20))
    
    # read the point cloud from the raw scan file
    current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32).reshape((-1, 4))
    
    # get range projection correspondences
    _, _, _, proj_idx = range_projection(current_vertex, max_range=np.inf)
    
    # init a semantic image array
    proj_prob = np.full((proj_H, proj_W, 20), -1,
                         dtype=np.float32)  # [H,W]: probs
    
    # fill in a semantic image
    proj_prob[proj_idx >= 0] = probs[proj_idx[proj_idx >= 0]]
    
    # generate the destination path
    base_name = os.path.basename(scan_paths[idx]).replace('.bin', '')
    dst_path = os.path.join(dst_folder, base_name)

    # save the semantic image as format of .npy
    np.save(dst_path, proj_prob)
    semantics.append(proj_prob)
    print('finished generating semantic data at: ', dst_path)

  return semantics


if __name__ == '__main__':
  semantic_folder = '/folder/of/semantic/probabilities'
  scan_folder = '/folder/of/lidar/scans'
  dst_folder = '/folder/to/store/semantic/data'
  
  semantic_data = gen_semantic_data(semantic_folder, scan_folder, dst_folder)
