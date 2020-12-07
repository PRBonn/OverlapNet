#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: A class used for inferring overlap and yaw-angle between two LiDAR scans
#        To get a log file
import importlib
import logging
import os
import sys

import keras
import numpy as np
import yaml

importlib.reload(logging)  # needed for ipython console

import generateNet
from ImagePairOverlapSequenceFeatureVolume import ImagePairOverlapSequenceFeatureVolume
from ImagePairOverlapOrientationSequence import ImagePairOverlapOrientationSequence


class Infer():
  """ A class used for inferring overlap and yaw-angle between LiDAR scans.
  """
  
  def __init__(self, config):
    """ Initialization of the class
        Args:
          config: A dict with configuration values, usually loaded from a yaml file
    """
    self.network_output_size = config['model']['leg_output_width']
    self.seq = config['infer_seqs']
    
    self.datasetpath = config['data_root_folder']
    
    if 'use_depth' in config:
      self.use_depth = config['use_depth']
    else:
      self.use_depth = True
    
    if 'use_normals' in config:
      self.use_normals = config['use_normals']
    else:
      self.use_normals = True
    
    if 'use_class_probabilities' in config:
      self.use_class_probabilities = config['use_class_probabilities']
    else:
      self.use_class_probabilities = False
    
    if 'use_class_probabilities_pca' in config:
      self.use_class_probabilities_pca = config['use_class_probabilities_pca']
    else:
      self.use_class_probabilities_pca = False
    
    if 'use_intensity' in config:
      self.use_intensity = config['use_intensity']
    else:
      self.use_intensity = False
    
    # no channels for input
    self.no_input_channels = 0
    if config['use_depth']:
      self.no_input_channels += 1
    if config['use_normals']:
      self.no_input_channels += 3
    if config['use_intensity']:
      self.no_input_channels += 1
    if config['use_class_probabilities']:
      if config['use_class_probabilities_pca']:
        self.no_input_channels += 3
      else:
        self.no_input_channels += 20
    
    # Input shape of model
    self.inputShape = config['model']['inputShape']
    if len(self.inputShape) == 3:
      pass
    elif len(self.inputShape) == 2:
      self.inputShape.append(self.no_input_channels)
    else:
      self.inputShape[2] = self.no_input_channels
    
    self.batch_size = config['batch_size']
    
    # %% Create two nets: leg and head
    legsType = config['model']['legsType']
    overlap_head = config['model']['overlap_head']
    orientation_head = config['model']['orientation_head']
    
    network_generate_method_leg = getattr(generateNet, 'generate' + legsType)
    network_generate_method_overlap_head = getattr(generateNet, 'generate' + overlap_head)
    network_generate_method_orientation_head = getattr(generateNet, 'generate' + orientation_head)
    # Input for leg
    leg_input_l = keras.Input(self.inputShape)
    leg_input_r = keras.Input(self.inputShape)
    
    # The leg: Note that only encoded_l is actually used
    (encoded_l, encoded_r) = network_generate_method_leg(
      leg_input_l, leg_input_r, self.inputShape, config['model'], False)
    self.leg = keras.Model(inputs=leg_input_l, outputs=encoded_l)
    
    # Input for head
    inputHead_l = keras.Input(encoded_l.shape[1:])
    inputHead_r = keras.Input(encoded_r.shape[1:])
    
    # The head
    overlap_predict = network_generate_method_overlap_head(inputHead_l, inputHead_r, config['model'])
    orientation_predict = network_generate_method_orientation_head(inputHead_l, inputHead_r, config['model'])
    
    self.head = keras.Model(inputs=(inputHead_l, inputHead_r), outputs=[overlap_predict, orientation_predict])
    
    # previous feature volumes
    self.feature_volumes = []
    
    # Load weights from training
    pretrained_weightsfilename = config['pretrained_weightsfilename']
    if len(pretrained_weightsfilename) > 0:
      self.leg.load_weights(pretrained_weightsfilename, by_name=True)
      self.head.load_weights(pretrained_weightsfilename, by_name=True)
    else:
      print('Pre-trained weights was not found in:', pretrained_weightsfilename)
  
  def infer_one(self, filepath1, filepath2):
    """ Infer with one input pair.
          Args:
            filepath1: path of LiDAR scan 1
            filepath2: path of LiDAR scan 2
            
          Returns:
            [overlap, yaw]
    """
    # check file format
    if not filepath1.endswith('.bin') or not filepath2.endswith('.bin'):
      raise Exception('Please check the LiDAR file format, '
                      'this implementation currently only works with .bin files.')
    
    filename1 = os.path.basename(filepath1).replace('.bin', '')
    filename2 = os.path.basename(filepath2).replace('.bin', '')
    self.filenames = np.array([filename2, filename1])
    
    # check preprocessed data
    preprocess_data_folder = os.path.join(self.datasetpath, self.seq)
    if not os.path.isdir(preprocess_data_folder):
      raise Exception('Please first generate preprocessed input data.')
    
    # generate leg feature_volumes
    feature_volumes = self.create_feature_volumes(self.filenames)
    
    indizes = np.zeros((1, 2), dtype=int)
    indizes[0, 0] = 0
    indizes[0, 1] = 1
    test_generator_head = ImagePairOverlapSequenceFeatureVolume(indizes, np.array([[0.]]),
                                                                self.batch_size, feature_volumes)
    model_outputs = self.head.predict_generator(test_generator_head, max_queue_size=10,
                                                workers=8, verbose=1)
    overlap_out = model_outputs[0][0]
    yaw_out = 180 - np.argmax(model_outputs[1], axis=1)
    
    return overlap_out, yaw_out
  
  def infer_multiple(self, current_frame_id, reference_frame_id):
    """ Infer for loopclosing: The current frame versus old frames.
        This is a special function, because only the feature volume of the current frame
        is computed. For the older reference frames the feature volumes must be already
        there. This is usually the case, because they were the "current frame" in
        previous calls of the function and the feature volumes are stored within
        this class for every call. For the starting frame, call this function
        with an empty list of reference_frame_id.
        
        For a more general usage use Infer.infer_multiple_vs_multiple().
    
        Args:
          current_frame_id: The id (an int) of the current frame. This corresponds
                            to depth and normal and scan files, 
                            e.g. 6 --> file 000006.bin or 000006.npy is used.
                            For this frame the feature volume is calculated and appended to
                            the list of already calculated feature volumes.
          reference_frame_id: a list of ids (aka a list of ints) of previous frames. 
                              Can be empty
        Returns:
          A tuple (overlaps, yaws) with two lists of the overlaps and yaw angles between the scans
    """
    filename = [str(current_frame_id).zfill(6)]
    self.feature_volumes.append(self.create_feature_volumes(filename)[0])
    
    if len(reference_frame_id) > 0:
      pair_indizes = np.zeros((len(reference_frame_id), 2), dtype=int)
      pair_indizes[:, 1] = np.ones(len(reference_frame_id)) * current_frame_id
      pair_indizes[:, 0] = reference_frame_id
      
      test_generator_head = ImagePairOverlapSequenceFeatureVolume(pair_indizes, np.zeros((len(pair_indizes))),
                                                                  self.batch_size, np.array(self.feature_volumes))
      model_outputs = self.head.predict_generator(test_generator_head, max_queue_size=10,
                                                  workers=8, verbose=1)
      
      overlap_out = model_outputs[0].squeeze()
      yaw_out = 180 - np.argmax(model_outputs[1], axis=1)
    
      return overlap_out, yaw_out
    
    else:
      return None
    
  def infer_multiple_vs_multiple(self, file_names, first_idxs, second_idxs):
    """ Infer with multiple input pairs.
        Args:
          file_names: All sample scan files, for example ['000000','000001','000004'] or ['000000.bin','000001.bin','000004.bin']
          first_idxs: indizes of the first LiDAR scans ,for example [0,1,2]
          second_idxs: indizes of the second LiDAR scans ,for example [2,1,1]
          Example:
            infer_multiple(['000000','000001','000004'],[0,1,2],[2,1,1])
            This will output the overlaps and yaws of ('000000','000004'),('000001','000001') and ('000004','000001')
        Returns:
            A tuple (overlaps, yaws) with two lists of the overlaps and yaw angles between the scans
    """
    if len(first_idxs) != len(second_idxs):
      raise Exception('Please make sure the first_idxs and second_idxs have the same size.')
    file_names=[os.path.basename(v).replace('.bin', '') for v in file_names]
    self.feature_volumes=self.create_feature_volumes(file_names)
    
    if len(second_idxs) > 0:
      pair_indizes = np.zeros((len(second_idxs), 2), dtype=int)
      pair_indizes[:, 1] = first_idxs
      pair_indizes[:, 0] = second_idxs
      
      test_generator_head = ImagePairOverlapSequenceFeatureVolume(pair_indizes, np.zeros((len(pair_indizes))),
                                                                  self.batch_size, np.array(self.feature_volumes))
      model_outputs = self.head.predict_generator(test_generator_head, max_queue_size=10,
                                                  workers=8, verbose=1)
      
      overlap_out = model_outputs[0].squeeze()
      yaw_out = 180 - np.argmax(model_outputs[1], axis=1)
    
      return overlap_out, yaw_out
    
    else:
      return None    
  
  def create_feature_volumes(self, filenames):
    """ create feature volumes, thus execute the leg.
        Args:
          filenames: numpy array of input file names (list of strings without extension, e.g. ['000000', '000001'])
        Returns:
          A n x width x height x channels numpy array of feature volumes
    """
    generator_leg = ImagePairOverlapOrientationSequence(self.datasetpath,
                                                        filenames, [],
                                                        [self.seq for _ in
                                                         range(len(filenames))], [],
                                                        np.zeros((len(filenames))),
                                                        np.zeros((len(filenames))),
                                                        self.network_output_size, self.batch_size,
                                                        self.inputShape[0], self.inputShape[1],
                                                        self.no_input_channels,
                                                        use_depth=self.use_depth,
                                                        use_normals=self.use_normals,
                                                        use_class_probabilities=self.use_class_probabilities,
                                                        use_intensity=self.use_intensity,
                                                        use_class_probabilities_pca=self.use_class_probabilities_pca)
    
    feature_volumes = self.leg.predict_generator(generator_leg, max_queue_size=10,
                                                 workers=8, verbose=1)
    
    return feature_volumes
    
# Test the infer functions
if __name__ == '__main__':
  configfilename = 'config/network.yml'
  
  if len(sys.argv) > 1:
    configfilename = sys.argv[1]
  if len(sys.argv) > 2:
    scan1 = sys.argv[2]
    scan2 = sys.argv[3]
  else:
    scan1 = '000020.bin'
    scan2 = '000021.bin'
  
  config = yaml.load(open(configfilename))
  infer = Infer(config)
  
  print('Test infer one ...')
  overlap, yaw = infer.infer_one(scan1, scan2)
  print("Overlap:  ", overlap)
  print("Orientation:  ", yaw)
  
  print('Test infer multiple (last scan vs previous) ...')
  # Note that this is special for loop-closing, all previous have to be asked !
  infer.infer_multiple(0, [])
  overlaps, yaws = infer.infer_multiple(1, [0])
  overlaps, yaws = infer.infer_multiple(2, [0,1])
  print("Overlaps:  ", overlaps)
  print("Orientations:  ", yaws)

  print('Test infer many versus many ...')
  file_names  = ['000010','000021','000022.bin']
  first_idxs  = [0,1,2]
  second_idxs = [2,1,1]
  overlaps, yaws = infer.infer_multiple_vs_multiple(file_names, first_idxs, second_idxs)
  print("Overlaps:  ", overlaps)
  print("Orientations:  ", yaws)
