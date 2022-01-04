#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a simple example to split the ground truth data into training and validation parts

import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(ground_truth_mapping):
  """Split the ground truth data into training and validation two parts.
     Args:
       ground_truth_mapping: the raw ground truth mapping array
     Returns:
       train_set: data used for training
       test_set: data used for validation
  """
  # set the ratio of validation data
  test_size = int(len(ground_truth_mapping) / 10)
  
  # use sklearn library to split the data
  train_set, test_set = train_test_split(ground_truth_mapping, test_size=test_size)

  print('finished generating training data and validation data')
  
  return train_set, test_set


if __name__ == '__main__':
  # read from npz file
  ground_truth_file = 'path/to/the/groun-truth/file'
  ground_truth_mapping = np.load(ground_truth_file)['arr_0']
  
  split_train_val(ground_truth_mapping)

