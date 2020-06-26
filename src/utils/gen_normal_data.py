#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a script to generate normal data

try: from utils import *
except: from utils import *


def gen_normal_data(scan_folder, dst_folder):
  """ Generate projected normal data in the shape of (64, 900, 3).
      The input raw data are in the shape of (Num_points, 3).
  """
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'normal')
  try:
    os.stat(dst_folder)
    print('generating normal data in: ', dst_folder)
  except:
    print('creating new normal folder: ', dst_folder)
    os.mkdir(dst_folder)
  
  # load LiDAR scan files
  scan_paths = load_files(scan_folder)
  normals = []
  # iterate over all scan files
  for idx in range(len(scan_paths)):
    # load a point cloud
    current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    
    # generate range image from point cloud
    proj_range, proj_vertex, _, _ = range_projection(current_vertex)

    # generate normal image
    normal_data = gen_normal_map(proj_range, proj_vertex)
    
    # generate the destination path
    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    
    # save the semantic image as format of .npy
    np.save(dst_path, normal_data)
    normals.append(normal_data)
    print('finished generating intensity data at: ', dst_path)
    
  return normals


if __name__ == '__main__':
  scan_folder = '/folder/of/lidar/scans'
  dst_folder = '/folder/to/store/normal/data'
  
  normal_data = gen_normal_data(scan_folder, dst_folder)
