#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This is an implementation of normalized cross correlation between two feature volumes
#        It is different to the normal convolution in neural networks. Instead of convolving data with a filter,
#        it convolves data with other data. For this reason, it has no trainable weights.
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.layers import Input
from keras.models import Model

from RangePadding2D import RangePadding2D


class NormalizedCorrelation2D(Layer):
  """ A layer which does normalized correlation (==convolution) of two given inputs.
      This is different from the standard convolution, as it is done with a kernel with weights
      and one input.
  """
      
  def __init__(self, output_dim=1, normalize='euclidean', **kwargs):
    """ Initialization as a layer.
        Args:
          normlize: a string which defines the normalization done:
                    'euclidean': just euclidian normalization
                    'scaling': scale, that result in in range 0..1
                    'standardization': Substract the mean and then do euclidean normalization
    """
    super(NormalizedCorrelation2D, self).__init__(**kwargs)
    self.output_dim = output_dim  # output size
    self.use_norm = normalize
  
  def build(self, input_shape):
    super(NormalizedCorrelation2D, self).build(input_shape)
  
  def compute_output_shape(self, input_shape):
    """ Just return the output shape.
    """
    return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.output_dim)
  
  def call(self, x, mask=None):
    """ The actual processing in the layer: Normalize, padd, then convolution.
    """
    input_1, input_2 = x
    input_shape = input_1._keras_shape
    
    assert input_shape == input_2._keras_shape
    
    self.H = input_shape[1]
    self.W = input_shape[2]
    self.C = input_shape[3]

    # normalization
    if self.use_norm is 'euclidean':
      input_1 = K.l2_normalize(input_1, axis=2)
      input_2 = K.l2_normalize(input_2, axis=2)

    if self.use_norm is 'scaling':
      input_1_min = K.min(input_1, axis=2, keepdims=True)
      input_1_max = K.max(input_1, axis=2, keepdims=True)
      input_1 = (input_1 - input_1_min) / (input_1_max - input_1_min + 0.000001)
  
      input_2_min = K.min(input_2, axis=2, keepdims=True)
      input_2_max = K.max(input_2, axis=2, keepdims=True)
      input_2 = (input_2 - input_2_min) / (input_2_max - input_2_min + 0.000001)

    if self.use_norm is 'standardization':
      input_1 = (input_1 - K.mean(input_1, axis=2, keepdims=True)) + 0.00001
      input_1 = K.l2_normalize(input_1, axis=2)
      input_2 = (input_2 - K.mean(input_2, axis=2, keepdims=True)) + 0.00001
      input_2 = K.l2_normalize(input_2, axis=2)

    # Pad the first input1 circular, so that a correlation can be computed for
    # every horizontal position    
    padding1 = RangePadding2D(padding=self.W // 2)(input_1)

    out = tf.scan(self.single_sample_corr,
                  elems=[padding1, input_2],
                  initializer=(K.zeros((int(self.H), int(self.W), int(self.output_dim))))
                  )
    return out
  
  def single_sample_corr(self, previous, features):
    """ Reformatting for a single sample.
    """
    
    fea1, fea2 = features  # fea1: the displacement, fea2: the kernel
    corr = self.correlation(fea1, fea2)
    
    corr = K.reshape(corr, (self.H, self.W, self.output_dim))
    
    return corr

  def correlation(self, displace, kernel):
    """ Do the actual convolution==correlation.
    """  
    # Given an input tensor of shape [batch, in_height, in_width, in_channels]
    displace = K.expand_dims(displace, 0)
    
    # a kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    kernel = K.expand_dims(kernel, 3)
    
    out = K.conv2d(displace, kernel, padding='valid', data_format='channels_last')
        
    out = K.squeeze(out, 0)
    
    return out


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  
  resolution = 6
  img1 = np.zeros((1, 6, 1))
  for idx in range(6):
    img1[0, idx, 0] = idx
  
  img2 = np.zeros((1, 6, 1))

  img2[0, 1:, :] = img1[0, :5, :]
  img2[0, 0, :] = img1[0, 5, :]
  
  img1 = np.expand_dims(img1, 0)
  img2 = np.expand_dims(img2, 0)
  
  print(img1.shape)
  print(img2.shape)
  
  input1 = Input(shape=(1, resolution, 1))
  input2 = Input(shape=(1, resolution, 1))
  
  norm_corr = NormalizedCorrelation2D(output_dim=1, normalize='euclidean')([input1, input2])
  
  model = Model([input1, input2], norm_corr)
  
  corr = model.predict([img1, img2])
  
  print(np.max(corr))
  plt.plot(corr[0, 0, :, 0], label='0')
  
  plt.legend()
  plt.show()
