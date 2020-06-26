#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a script to generate depth data

try: from utils import *
except: from utils import *


def gen_depth_data(scan_folder, dst_folder, normalize=False):
  """ Generate projected range data in the shape of (64, 900, 1).
      The input raw data are in the shape of (Num_points, 3).
  """
  # specify the goal folder
  dst_folder = os.path.join(dst_folder, 'depth')
  try:
    os.stat(dst_folder)
    print('generating depth data in: ', dst_folder)
  except:
    print('creating new depth folder: ', dst_folder)
    os.mkdir(dst_folder)
  
  # load LiDAR scan files
  scan_paths = load_files(scan_folder)

  depths = []
  
  # iterate over all scan files
  for idx in range(len(scan_paths)):
    # load a point cloud
    current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    
    proj_range, _, _, _ = range_projection(current_vertex)
    
    # normalize the image
    if normalize:
      proj_range = proj_range / np.max(proj_range)
    
    # generate the destination path
    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    
    # save the semantic image as format of .npy
    np.save(dst_path, proj_range)
    depths.append(proj_range)
    print('finished generating depth data at: ', dst_path)

  return depths
  

if __name__ == '__main__':
  scan_folder = '/folder/of/lidar/scans'
  dst_folder = '/folder/to/store/depth/data'
  
  depth_data = gen_depth_data(scan_folder, dst_folder)
