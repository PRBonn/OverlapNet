#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: A custom keras padding layer.
#        pad([1 2 3 4], 2) -> [3, 4, 1, 2, 3, 4, 1]
import numpy as np
from keras.layers import Layer
from keras.models import Sequential
import keras.backend as K


class RangePadding2D(Layer):
  """ A keras layer which does horizontal padding. The input tensor
      is padded in the width direction.
  """
  
  def __init__(self, padding, **kwargs):
    """ Initialization of the layer.
        Args:
          padding: defines how much will be padded "before" and "after" the input.
                   The input is padded in width direction like this:
                     [ padding:end, original, beginning:padding-1]
                   Usually one uses half of the width for this argument to have a symmetric padding
    """                   
    self.padding = padding
    super(RangePadding2D, self).__init__(**kwargs)
  
  def build(self, input_shape):
    super(RangePadding2D, self).build(input_shape)
  
  def call(self, inputs):
    if K.backend() == "tensorflow":
      # only do range padding in width dimension
      out = K.concatenate([inputs[:, :, self.padding:, :], inputs[:, :, :, :], inputs[:, :, :self.padding-1, :]],
                          axis=2)
    else:
      raise Exception("Backend " + K.backend() + "not implemented")
    return out
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1], input_shape[2] + input_shape[2] - 1 , input_shape[3])


if __name__ == "__main__":
  one_channel_test = True
  two_channel_test = True
  
  if two_channel_test:
    # set test data
    image_raw = [[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]]
    
    image = np.expand_dims(np.expand_dims(np.array(image_raw), 0), 0)
    print("input image shape: ", image.shape)
    print("Input:")
    print(image)
    
    # build Keras model
    model = Sequential()
    rlayer = RangePadding2D(padding=3, input_shape=(1, 6, 2))
    model.add(rlayer)
    model.build()
    
    # simply apply existing filter, we use predict with no training
    out = model.predict(image)
    print("Output shape: ",out.shape)
    print("result of compute_output_shape (should be the same):", rlayer.compute_output_shape(rlayer.input_shape))
    print("Output:")
    print(out)
  
  if one_channel_test:
    # one channel test
    image_raw = [[1, 2, 3, 4, 5, 6]]
    
    # pad to channels_last format
    # which is [batch, width, height, channels]=[1,1,6,1]
    image = np.expand_dims(np.expand_dims(np.array(image_raw), 2), 0)
    print("input image shape: ", image.shape)
    print("Input:")
    print(image)
    
    # build Keras model
    model = Sequential()
    rlayer = RangePadding2D(padding=3, input_shape=(1, 6, 1))
    model.add(rlayer)
    model.build()
    
    # simply apply existing filter, we use predict with no training
    out = model.predict(image)
    print("Output shape: ",out.shape)
    print("result of compute_output_shape (should be the same):", rlayer.compute_output_shape(rlayer.input_shape))
    print("Output:")
    print(out)
