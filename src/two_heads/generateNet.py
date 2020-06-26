#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: Generate a neural net for overlap detection.
import sys

import keras.backend as backend
import keras.optimizers
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Lambda
from keras.models import Model
from keras.regularizers import l2
from NormalizedCorrelation2D import NormalizedCorrelation2D


def DeltaLayer(encoded_l, encoded_r, negateDiffs=False):
  """
  A Layer which computes all possible absolute differences of
  all pixels. Input are two feature volumes, e.g. result of a conv layer
  Hints:
  - The Reshape reshapes a matrix row-wise, that means,

    Reshape( (6,1) ) ([ 1 2 3
                      4 5 6]) is

                      1
                      2
                      3
                      4
                      5
                      6
  - Algorithm:
    - The left  leg is reshaped to a w*h x 1  column vector (for each channel)
    - The right leg is reshaped to a  1 x w*h row vector (for each channel)
    - The left is tiled along colum axis, so from w*h x 1 to w*h x w*h (per channel)
    - The right is tiled along row axis, so from 1 x w*h to w*h x w*h
    - The absolute difference is calculated
  Args:
      encoded_l, encoded_r : left and right image tensor (batchsize,w,h,channels)
                             must have same size
      negateDiffs: if True then not abs(diffs), but -abs(diffs) is returned.
                   Default: False
  Returns:
      difference tensor, has size (batchsize, w*h, w*h, channels)
  """
  w = encoded_l.shape[1].value
  h = encoded_l.shape[2].value
  chan = encoded_l.shape[3].value
  reshapel = Reshape((w * h, 1, chan))
  reshaped_l = reshapel(encoded_l)
  reshaper = Reshape((1, w * h, chan))
  reshaped_r = reshaper(encoded_r)
  
  tiled_l = Lambda(lambda x: backend.tile(x, [1, 1, w * h, 1]))(reshaped_l)
  tiled_r = Lambda(lambda x: backend.tile(x, [1, w * h, 1, 1]))(reshaped_r)
  
  if negateDiffs:
    diff = Lambda(lambda x: -backend.abs(x[0] - x[1]))([tiled_l, tiled_r])
  else:
    diff = Lambda(lambda x: backend.abs(x[0] - x[1]))([tiled_l, tiled_r])
  
  return diff


def generateDeltaLayerConv1NetworkHead(encoded_l, encoded_r, config={}):
  """
  Generate Head of DeltaLayerConv1Network.
  Args:
    encoded_l, encoded_r: the feature volumes of the two images,
                          thus the last tensor of the leg
    config: dictionary of configuration parameters, usually from a yaml file
            All keys have default arguments, so they need not to be present
            Current parameters:
                conv1NetworkHead_conv1size: The size s of the 1xs and
                                            sx1 convolutions directly done
                                            for the DeltaLayer (which computes
                                            all possible differences of the
                                            feature volumnes). Default:
                                            15 (to be consistent with old
                                            versions of this code).
                                            This size should be actually the
                                            number of columns of the feature
                                            volume.
  Returns:
    the final tensor of the head which is 1x1, the overlap percentage
    0.0-1.0
  """
  # default parameters
  if not 'conv1NetworkHead_conv1size' in config:
    config['conv1NetworkHead_conv1size'] = 15

  # combine the two legs
  diff = DeltaLayer(encoded_l, encoded_r)

  # densify the information across feature maps
  kernel_regularizer = None
  combinedconv1 = Conv2D(64, (1, config['conv1NetworkHead_conv1size']),
                         strides=(1, config['conv1NetworkHead_conv1size']),
                         activation='linear',
                         kernel_regularizer=kernel_regularizer, name="c_conv1")
  combined2 = combinedconv1(diff)

  combinedconv2 = Conv2D(128, (config['conv1NetworkHead_conv1size'], 1),
                         strides=(config['conv1NetworkHead_conv1size'], 1),
                         activation='relu',
                         kernel_regularizer=kernel_regularizer, name="c_conv2")
  combined3 = combinedconv2(combined2)

  combinedconv3 = Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=kernel_regularizer, name="c_conv3")
  combined4 = combinedconv3(combined3)

  flattened = Flatten()(combined4)

  prediction = Dense(1, activation='sigmoid', name='overlap_output')(flattened)

  return prediction


def generate360OutputkLegs(left_input, right_input, input_shape=(50, 50, 1), config={},
                           smallNet=False):
  """
  Generate legs like in the DeltaLayerConv1Network.
  Here we use several Conv2D layer to resize the output of leg into 360

  Args:
    input_shape: A tupel with three elements which is the size of the input images.
    left_input, right_input: Two tensors of size input_shape which define the input
                             of the two legs
    config: dictionary of configuration parameters, usually from a yaml file
            All keys have default arguments, so they need not to be present
            Current parameters:
            - strides_layer1: Stride of first conv layer, default (2,2)
            - additional_unsymmetric_layer3a: Boolean. If true an additional layer 3a
                                  will be added with stride(1,2) after layer 3
                                  Default: False
    smallNet: a boolean. If true, a very tiny net is defined. Default: False

  Returns:
    a tuple with two tensors: the left and right feature volume
  """
  
  # default values for configuration parameters
  if not 'strides_layer1' in config:
    config['strides_layer1'] = (2, 2)
  if not 'additional_unsymmetric_layer3a' in config:
    config['additional_unsymmetric_layer3a'] = False
  
  # build convnet to use in each siamese 'leg'
  if (smallNet):
    finalconv = Conv2D(2, (5, 15), activation='relu',
                       padding='valid', strides=5, input_shape=input_shape,
                       name='s_conv1', kernel_regularizer=l2(2e-4))
    l = finalconv(left_input)
    r = finalconv(right_input)
    return (l, r)
  else:
    
    # kernel_regularizer=l2(1e-8)
    kernel_regularizer = None
    # conv1=Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
    conv1 = Conv2D(16, (5, 15), strides=config['strides_layer1'], activation='relu', input_shape=input_shape,
                   kernel_regularizer=kernel_regularizer, name="s_conv1")
    l = conv1(left_input)
    r = conv1(right_input)
    
    # conv2=Conv2D(128, (7, 7), activation='relu',
    conv2 = Conv2D(32, (3, 15), strides=(2, 1), activation='relu',
                   kernel_regularizer=kernel_regularizer, name="s_conv2")
    l = conv2(l)
    r = conv2(r)
    
    # conv3=Conv2D(128, (4, 4), activation='relu',
    conv3 = Conv2D(64, (3, 15), strides=(2, 1), activation='relu',
                   kernel_regularizer=kernel_regularizer, name="s_conv3")
    l = conv3(l)
    r = conv3(r)
    
    if config['additional_unsymmetric_layer3a']:
      conv3a = Conv2D(64, (3, 12), strides=(2, 1), activation='relu',
                      kernel_regularizer=kernel_regularizer, name="s_conv3a")
      l = conv3a(l)
      r = conv3a(r)
    
    conv4 = Conv2D(128, (2, 9), strides=(2, 1), activation='relu', name="s_conv4",
                   kernel_regularizer=kernel_regularizer)
    l = conv4(l)
    r = conv4(r)
    
    conv5 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv5",
                   kernel_regularizer=kernel_regularizer)
    l = conv5(l)
    r = conv5(r)
    
    conv6 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv6",
                   kernel_regularizer=kernel_regularizer)
    l = conv6(l)
    r = conv6(r)
    
    conv7 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv7",
                   kernel_regularizer=kernel_regularizer)
    l = conv7(l)
    r = conv7(r)
    
    conv8 = Conv2D(128, (1, 7), strides=(1, 1), activation='relu', name="s_conv8",
                   kernel_regularizer=kernel_regularizer)
    l = conv8(l)
    r = conv8(r)
    
    conv9 = Conv2D(128, (1, 5), strides=(1, 1), activation='relu', name="s_conv9",
                   kernel_regularizer=kernel_regularizer)
    l = conv9(l)
    r = conv9(r)
    
    conv10 = Conv2D(128, (1, 3), strides=(1, 1), activation='relu', name="s_conv10",
                    kernel_regularizer=kernel_regularizer)
    l = conv10(l)
    r = conv10(r)
    
    return (l, r)


def generate360OutputkLegsFixed(left_input, right_input, input_shape=(50, 50, 1), config={},
                                smallNet=False):
  """
  Generate legs like in the DeltaLayerConv1Network.
  Here we use several Conv2D layer to resize the output of leg into 360.
  This version of the leg sets all layers to not_trainable, thus the weights
  will not be changed during training.

  Args:
    input_shape: A tupel with three elements which is the size of the input images.
    left_input, right_input: Two tensors of size input_shape which define the input
                             of the two legs
    config: dictionary of configuration parameters, usually from a yaml file
            All keys have default arguments, so they need not to be present
            Current parameters:
            - strides_layer1: Stride of first conv layer, default (2,2)
            - additional_unsymmetric_layer3a: Boolean. If true an additional layer 3a
                                  will be added with stride(1,2) after layer 3
                                  Default: False
    smallNet: a boolean. If true, a very tiny net is defined. Default: False
    
  Returns:
    a tuple with two tensors: the left and right feature volume
  """

  # default values for configuration parameters
  if not 'strides_layer1' in config:
    config['strides_layer1'] = (2, 2)
  if not 'additional_unsymmetric_layer3a' in config:
    config['additional_unsymmetric_layer3a'] = False

  # build convnet to use in each siamese 'leg'
  if (smallNet):
    finalconv = Conv2D(2, (5, 15), activation='relu',
                       padding='valid', strides=5, input_shape=input_shape,
                       name='s_conv1', kernel_regularizer=l2(2e-4), trainable=False)
    l = finalconv(left_input)
    r = finalconv(right_input)
    return (l, r)
  else:
  
    # kernel_regularizer=l2(1e-8)
    kernel_regularizer = None
    # conv1=Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
    conv1 = Conv2D(16, (5, 15), strides=config['strides_layer1'], activation='relu', input_shape=input_shape,
                   kernel_regularizer=kernel_regularizer, name="s_conv1", trainable=False)
    l = conv1(left_input)
    r = conv1(right_input)
  
    # conv2=Conv2D(128, (7, 7), activation='relu',
    conv2 = Conv2D(32, (3, 15), strides=(2, 1), activation='relu',
                   kernel_regularizer=kernel_regularizer, name="s_conv2", trainable=False)
    l = conv2(l)
    r = conv2(r)
  
    # conv3=Conv2D(128, (4, 4), activation='relu',
    conv3 = Conv2D(64, (3, 15), strides=(2, 1), activation='relu',
                   kernel_regularizer=kernel_regularizer, name="s_conv3", trainable=False)
    l = conv3(l)
    r = conv3(r)
  
    if config['additional_unsymmetric_layer3a']:
      conv3a = Conv2D(64, (3, 12), strides=(2, 1), activation='relu',
                      kernel_regularizer=kernel_regularizer, name="s_conv3a", trainable=False)
      l = conv3a(l)
      r = conv3a(r)
  
    conv4 = Conv2D(128, (2, 9), strides=(2, 1), activation='relu', name="s_conv4",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv4(l)
    r = conv4(r)
  
    conv5 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv5",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv5(l)
    r = conv5(r)
  
    conv6 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv6",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv6(l)
    r = conv6(r)
  
    conv7 = Conv2D(128, (1, 9), strides=(1, 1), activation='relu', name="s_conv7",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv7(l)
    r = conv7(r)
  
    conv8 = Conv2D(128, (1, 7), strides=(1, 1), activation='relu', name="s_conv8",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv8(l)
    r = conv8(r)
  
    conv9 = Conv2D(128, (1, 5), strides=(1, 1), activation='relu', name="s_conv9",
                   kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv9(l)
    r = conv9(r)
  
    conv10 = Conv2D(128, (1, 3), strides=(1, 1), activation='relu', name="s_conv10",
                    kernel_regularizer=kernel_regularizer, trainable=False)
    l = conv10(l)
    r = conv10(r)
  
    return (l, r)

  
def generateCorrelationHead(encoded_l, encoded_r, config={})  :
    """
      Generate a head which does correlation.
      
    Args:
      encoded_l, encoded_r: the feature volumes of the two images,
                            thus the last tensor of the leg. Must be of size
                            batch_idx x 1 x no_col x no_channels
  
      config: dictionary of configuration parameters, usually from a yaml file
              Current parameters used here:
    
  
    Returns:
      the output feature (volume) of the head. Here: A feature volume with size 1XMx1
    """
    norm_corr = NormalizedCorrelation2D(output_dim=1, normalize='none')([encoded_l, encoded_r])

    # # show the gradient
    # from GradientPrint import GradientPrint
    # norm_corr = GradientPrint()([norm_corr, encoded_l])

    # from NonMaxSuppression import NonMaxSuppression
    # non_max = NonMaxSuppression(output_dim=1)(norm_corr)
    #
    norm_corr = Flatten(name='orientation_output')(norm_corr)
    
    return norm_corr


def generateSiameseNetworkTemplate(input_shape=(50, 50, 1), config={}, smallNet=False):
  """
  Generate a siamese network for overlap detection. Which legs and which
  head is used will be given in the config parameter.
  
  Args:
    input_shape: A tupel with three elements which is the size of the input images.
    config: dictionary of configuration parameters, usually from a yaml file
            Current parameters used here:
            legsType: name of the function (without "generate") for the legs
            headType: name of the function (without "generate") for the heads
            The config is given to the head and legs, so additional parameters can
            be given.

  Returns:
    the neural net as a keras.model
  """
  
  # Define the input
  left_input = Input(input_shape)
  right_input = Input(input_shape)
  
  # The two legs
  leg_method = getattr(sys.modules[__name__], 'generate' + config['legsType'])
  (encoded_l, encoded_r) = leg_method(
    left_input, right_input, input_shape,
    config, smallNet)
  
  # The overlap head
  head_method = getattr(sys.modules[__name__], 'generate' + config['overlap_head'])
  prediction_overlap = head_method(encoded_l, encoded_r, config)
  # prediction=generateDeltaLayerReLuNetworkHead(encoded_l, encoded_r)
  
  # The orientation head
  head_method = getattr(sys.modules[__name__], 'generate' + config['orientation_head'])
  prediction_orientation = head_method(encoded_l, encoded_r, config)
  
  # Generate a keras model out of the input and output tensors
  siamese_net = Model(inputs=[left_input, right_input], outputs=[prediction_overlap, prediction_orientation])
  return siamese_net


def generateSiameseNetworkTemplateLegs(left_input, right_input, input_shape=(50, 50, 1), config={},
                                       smallNet=False):
  """
  This is the Head for the generic generateSiameseNetworkTemplate. This
  is only a wrapper to be compatible with the naming of the function
  in validation_leg_head_separate.py. 
  Which legs are used will be given in the config parameter.
  
  Args:
    input_shape: A tupel with three elements which is the size of the input images.
    left_input, right_input: Two tensors of size input_shape which define the input
                             of the two legs

    config: dictionary of configuration parameters, usually from a yaml file
            Current parameters used here:
            legsType: name of the function (without "generate") for the legs
            The config is given to the head and legs, so additional parameters can
            be given.

  Returns:
    the neural net as a keras.model
  """
  leg_method = getattr(sys.modules[__name__], 'generate' + config['legsType'])
  (encoded_l, encoded_r) = leg_method(
    left_input, right_input, input_shape,
    config, smallNet)
  return (encoded_l, encoded_r)


def generateSiameseNetworkTemplateHead(encoded_l, encoded_r, config={}):
  """
  This is the Head for the generic generateSiameseNetworkTemplate. This
  is only a wrapper to be compatible with the naming of the function
  in validation_leg_head_separate.py. Which
  head is used will be given in the config parameter.
  
  Args:
    encoded_l, encoded_r: the feature volumes of the two images,
                          thus the last tensor of the leg
  
    config: dictionary of configuration parameters, usually from a yaml file
            Current parameters used here:
            headType: name of the function (without "generate") for the heads
            The config is given to the head and legs, so additional parameters can
            be given.

  Returns:
    the neural net as a keras.model
  """
  head_method = getattr(sys.modules[__name__], 'generate' + config['headType'])
  prediction = head_method(encoded_l, encoded_r, config)
  
  return prediction


# For testing/debuging
if __name__ == "__main__":
  input_shape = (64, 900, 16)
  config = {}
  config['legsType'] = '360OutputkLegs_smaller'
  config['overlap_head'] = 'DeltaLayerConv1NetworkHead'
  config['orientation_head'] = 'CorrelationHead'
  
  config['additional_unsymmetric_layer3a'] = True
  config['strides_layer1'] = [2, 2]
  model = generateSiameseNetworkTemplate(input_shape, config)
  optimizer = keras.optimizers.SGD(
    lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss="mean_squared_error", optimizer=optimizer)
  model.summary()
