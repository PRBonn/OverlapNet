#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: some utilities
import os
import math
import numpy as np


def load_poses(pose_path):
  """ Load ground truth poses (T_w_cam0) from file.
      Args: 
        pose_path: (Complete) filename for the pose file
      Returns: 
        A numpy array of size nx4x4 with n poses as 4x4 transformation 
        matrices
  """
  # Read and parse the poses
  poses = []
  try:
    if '.txt' in pose_path:
      with open(pose_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
          T_w_cam0 = T_w_cam0.reshape(3, 4)
          T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
          poses.append(T_w_cam0)
    else:
      poses = np.load(pose_path)['arr_0']
  
  except FileNotFoundError:
    print('Ground truth poses are not avaialble.')
  
  return np.array(poses)


def load_calib(calib_path):
  """ Load calibrations (T_cam_velo) from file.
  """
  # Read and parse the calibrations
  T_cam_velo = []
  try:
    with open(calib_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        if 'Tr:' in line:
          line = line.replace('Tr:', '')
          T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
          T_cam_velo = T_cam_velo.reshape(3, 4)
          T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
  
  except FileNotFoundError:
    print('Calibrations are not avaialble.')
  
  return np.array(T_cam_velo)


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_range, proj_vertex, proj_intensity, proj_idx


def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
  """ Generate a normal image given the range projection of a point cloud.
      Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
      Returns: 
        normal_data: each pixel contains the corresponding normal
  """
  normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
  
  # iterate over all pixels in the range image
  for x in range(proj_W):
    for y in range(proj_H - 1):
      p = current_vertex[y, x][:3]
      depth = current_range[y, x]
      
      if depth > 0:
        wrap_x = wrap(x + 1, proj_W)
        u = current_vertex[y, wrap_x][:3]
        u_depth = current_range[y, wrap_x]
        if u_depth <= 0:
          continue
        
        v = current_vertex[y + 1, x][:3]
        v_depth = current_range[y + 1, x]
        if v_depth <= 0:
          continue
        
        u_norm = (u - p) / np.linalg.norm(u - p)
        v_norm = (v - p) / np.linalg.norm(v - p)
        
        w = np.cross(v_norm, u_norm)
        norm = np.linalg.norm(w)
        if norm > 0:
          normal = w / norm
          normal_data[y, x] = normal
  
  return normal_data


def wrap(x, dim):
  """ Wrap the boarder of the range image.
  """
  value = x
  if value >= dim:
    value = (value - dim)
  if value < 0:
    value = (value + dim)
  return value


def euler_angles_from_rotation_matrix(R):
  """ From the paper by Gregory G. Slabaugh,
      Computing Euler angles from a rotation matrix
      psi, theta, phi = roll pitch yaw (x, y, z)
      Args:
        R: rotation matrix, a 3x3 numpy array
      Returns:
        a tuple with the 3 values psi, theta, phi in radians
  """
  
  def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)
  
  phi = 0.0
  if isclose(R[2, 0], -1.0):
    theta = math.pi / 2.0
    psi = math.atan2(R[0, 1], R[0, 2])
  elif isclose(R[2, 0], 1.0):
    theta = -math.pi / 2.0
    psi = math.atan2(-R[0, 1], -R[0, 2])
  else:
    theta = -math.asin(R[2, 0])
    cos_theta = math.cos(theta)
    psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
    phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
  return psi, theta, phi


def load_vertex(scan_path):
  """ Load 3D points of a scan. The fileformat is the .bin format used in
      the KITTI dataset.
      Args: 
        scan_path: the (full) filename of the scan file
      Returns: 
        A nx4 numpy array of homogeneous points (x, y, z, 1).
  """
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex


def load_files(folder):
  """ Load all files in a folder and sort.
  """
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths


semantic_mapping = {  # bgr
  0:  [0, 0, 0],          # "unlabeled", and others ignored
  1:  [245, 150, 100],    # "car"
  2:  [245, 230, 100],    # "bicycle"
  3:  [150, 60, 30],      # "motorcycle"
  4:  [180, 30, 80],      # "truck"
  5:  [255, 0, 0],        # "other-vehicle"
  6:  [30, 30, 255],      # "person"
  7:  [200, 40, 255],     # "bicyclist"
  8:  [90, 30, 150],      # "motorcyclist"
  9:  [255, 0, 255],      # "road"
  10: [255, 150, 255],    # "parking"
  11: [75, 0, 75],        # "sidewalk"
  12: [75, 0, 175],       # "other-ground"
  13: [0, 200, 255],      # "building"
  14: [50, 120, 255],     # "fence"
  15: [0, 175, 0],        # "vegetation"
  16: [0, 60, 135],       # "trunk"
  17: [80, 240, 150],     # "terrain"
  18: [150, 240, 255],    # "pole"
  19: [0, 0, 255]         # "traffic-sign"
}