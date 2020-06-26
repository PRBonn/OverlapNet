#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a script to generate intensity data

try: from utils import *
except: from utils import *


def gen_intensity_data(scan_folder, dst_folder):
  """ Generate projected intensity data in the shape of (64, 900, 1).
      The input raw data are in the shape of (Num_points, 1).
  """
  # specify the goal paths
  dst_folder = os.path.join(dst_folder, 'intensity')
  try:
    os.stat(dst_folder)
    print('creating intensity data in: ', dst_folder)
  except:
    print('creating new intensity folder: ', dst_folder)
    os.mkdir(dst_folder)
  
  # load LiDAR scan files
  scan_paths = load_files(scan_folder)
  intensities = []
  # iterate over all scan files
  for idx in range(len(scan_paths)):
    # load a point cloud
    current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    
    # generate intensity image from point cloud
    _, _, proj_intensity, _ = range_projection(current_vertex)
    
    # generate the destination path
    dst_path = os.path.join(dst_folder, str(idx).zfill(6))
    
    # save the semantic image as format of .npy
    np.save(dst_path, proj_intensity)
    intensities.append(proj_intensity)
    print('finished generating intensity data at: ', dst_path)
    
  return intensities


if __name__ == '__main__':
  scan_folder = '/folder/of/lidar/scans'
  dst_folder = '/folder/to/store/intensity/data'
  
  intensity_data = gen_intensity_data(scan_folder, dst_folder)
