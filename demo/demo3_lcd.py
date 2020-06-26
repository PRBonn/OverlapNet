#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a demo to show the loop closure detection results combined with pose uncertainty

import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/two_heads'))
from utils import *
from infer import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import animation

class AnimatedLCD(object):
  def __init__(self, configfilename, gt_poses, covs):
    """initialize the animation and data needed"""
    # Init the network
    config = yaml.load(open(configfilename))
    self.infer = Infer(config)
    
    # Setup data stream
    self.poses = gt_poses
    self.covs = covs
    self.stream = self.data_stream()
    
    # Setup the figure and axes
    self.fig, self.ax = plt.subplots()
    
    # set size
    self.point_size = 50
    self.offset = 5  # meters
    self.map_size = [min(gt_poses[:, 0, 3]) - self.offset,
                     max(gt_poses[:, 0, 3]) + self.offset,
                     min(gt_poses[:, 1, 3]) - self.offset,
                     max(gt_poses[:, 1, 3]) + self.offset]
    
    # Then setup FuncAnimation.
    self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.01,
                                       init_func=self.setup_plot, blit=True, save_count=len(gt_poses) - 1)
    # set the map
    self.ax.set(xlim=self.map_size[:2], ylim=self.map_size[2:])
    self.ax.plot(gt_poses[:, 0, 3], gt_poses[:, 1, 3], '--', alpha=0.5, c='black')
    self.traj_length = []
    self.inactive_time_thres = 100
    self.inactive_dist_thres = 50
    self.overlap_thres = 0.3

    # # save demo as video
    # self.ani.save('demo3.mp4', writer='imagemagick', dpi=150)
  
  def setup_plot(self):
    """Initialize plots"""
    # setup ax0
    self.ax_traj, = self.ax.plot([], [], c='b', label='trajectory')
    self.loop_closure = self.ax.scatter([], [], s=self.point_size, c="r", label='loops')
    self.ellipse = self.ax.add_patch(Ellipse((0, 0), 0, 0, linewidth=1, edgecolor='g', facecolor='none'))
    
    self.ax.axis('square')
    self.ax.set(xlim=self.map_size[:2], ylim=self.map_size[2:])
    self.ax.set_xlabel('X [m]')
    self.ax.set_ylabel('Y [m]')
    self.ax.set_title('Demo 3: Loop Closure Detection')
    # define legend
    search_space_legend = Line2D([0], [0], marker='o', color='w', markeredgecolor='g',
                                 markerfacecolor=None, markersize=10, label='search space')
    trajectory_legend = Line2D([0], [0], color='b', lw=2, label='trajectory')
    loop_legend = Line2D([0], [0], marker='o', color='w',
                         markerfacecolor='r', markersize=10, label='loops',)
    self.ax.legend(loc="lower right", handles=[trajectory_legend, search_space_legend, loop_legend])
    
    # combine all artists
    self.patches = [self.ax_traj, self.ellipse, self.loop_closure]
    
    return self.patches
  
  def get_predictions(self, idx, traj, ellipse):
    """Get overlapnet predictions with pose uncertainty"""
    # search only in previously frames and inactive map
    if idx < self.inactive_time_thres:
      self.infer.infer_multiple(idx, [])
      return None
    
    indices = np.arange(idx - self.inactive_time_thres)
    
    dist_delta = self.traj_length[idx] - np.array(self.traj_length)[indices]
    indices = indices[dist_delta > self.inactive_dist_thres]
    
    if len(indices) < 0:
      return None
    
    # check whether the prediction is in the search space or not
    angle = ellipse.angle
    width = ellipse.width
    height = ellipse.height
    
    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))
    
    xc = traj[idx, 0] - traj[indices, 0]
    yc = traj[idx, 1] - traj[indices, 1]
    
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = (xct ** 2 / (width / 2.) ** 2) + (yct ** 2 / (height / 2.) ** 2)
    
    reference_idx = indices[rad_cc < 1]
    
    if len(reference_idx) > 0:
      overlaps, _ = self.infer.infer_multiple(idx, reference_idx)
      if np.max(overlaps) > self.overlap_thres:
        return reference_idx[np.argmax(overlaps)]
    else:
      self.infer.infer_multiple(idx, [])
      return None
  
  def get_cov_ellipse(self, cov, center, nstd, **kwargs):
    """Return a matplotlib Ellipse patch representing the covariance matrix
       cov centred at centre and scaled by the factor nstd (n times the standard deviation).
    """
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)
    
    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals[:2])
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)
  
  def data_stream(self):
    """Data stream generator"""
    for idx in range(len(self.poses) - 1):
      traj = self.poses[:idx + 1, :2, 3]
      cov = covs[idx].reshape((6, 6))
      yield idx, traj, cov
  
  def update(self, i):
    """Update plots"""
    idx, traj, cov = next(self.stream)
    
    # set trajectories
    self.ax_traj.set_data(traj[:idx + 1, 0], traj[:idx + 1, 1])
    if idx > 0:
      dist_delta = np.linalg.norm(traj[idx] - traj[idx - 1])
      self.traj_length.append(self.traj_length[-1] + dist_delta)
    else:
      self.traj_length.append(0)
    
    # set search spaces
    ellipse = self.get_cov_ellipse(cov[:2, :2], traj[idx], 3,
                                   linewidth=1, edgecolor='r', facecolor='none')
    self.ellipse.center = ellipse.center
    self.ellipse.angle = ellipse.angle
    self.ellipse.width = ellipse.width
    self.ellipse.height = ellipse.height
    
    # set loop closures
    loop_closures_idx = self.get_predictions(idx, traj, ellipse)
    if loop_closures_idx is not None:
      self.loop_closure.set_offsets(traj[loop_closures_idx])
    
    # We need to return the updated artist for FuncAnimation to draw..
    # Note that it expects a sequence of artists, thus the trailing comma.
    return self.patches


if __name__ == '__main__':
  config_filename = 'config/demo.yml'
  
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  # load the configuration file
  config = yaml.load(open(config_filename))
  
  # set the related parameters
  covariance_file = config['Demo3']['covariance_file']
  poses_file = config['Demo3']['poses_file']
  calib_file = config['Demo3']['calib_file']
  scan_folder = config['Demo3']['scan_folder']
  config_file = config['Demo3']['network_config']
  
  # load raw LiDAR scan paths
  scan_paths = load_files(scan_folder)
  
  # load calibration parameter from the dataset
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)

  # load poses from either the dataset or SLAM or odometry methods
  poses = load_poses(poses_file)
  pose_0 = poses[0]
  inv_pose_0 = np.linalg.inv(pose_0)
  
  # for KITTI dataset, we need to convert the provided poses 
  # from the camera coordinate system into the LiDAR coordinate system  
  pose_new = []
  for pose in poses:
    pose_new.append(T_velo_cam.dot(inv_pose_0).dot(pose).dot(T_cam_velo))
  pose_new = np.array(pose_new)
  
  # load pose covariances generated from the SLAM or odometry methods
  covs = open(covariance_file)
  covs = [overlap.replace('\n', '').split() for overlap in covs.readlines()]
  covs = np.asarray(covs, dtype=float)
  
  # start the demo animation
  demo3 = AnimatedLCD(config_file, pose_new, covs)
  
  plt.show()
