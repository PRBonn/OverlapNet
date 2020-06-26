#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# A keras generator which generates batches out of cached feature volumes

import os
from keras.utils import Sequence
import numpy as np


class FeatureVolumeCacheSequence(Sequence):
  """ Initialize the dataset. It is assumed that all feature volumes
      are present. No ground truth is used, thus this class is only for inference.
      Args:
        config: struct with configuration:
            Used attributes:
              batch_size: size of a batch
              'data_root_folder','training_seqs': for path to feature volumes.
        feature_volume_size: a tuple with size of the feature volume (heightxwidthxchannels)
        cache_size: number of feature volumes to be stored (in CPU memory)
  """
  
  def __init__(self, config, feature_volume_size, cache_size):
    self.datasetpath = config['data_root_folder']+'/'+config['training_seqs']
    self.featurepath = self.datasetpath+'/feature_volumes/'
    self.batch_size = config['batch_size']
    self.feature_volume_size=feature_volume_size
    self.cache_size=cache_size
    
    # The cache:: feature volumes as
    # a numpy array with dimension n x w x h x chans
    self.cache=np.zeros( (cache_size,  feature_volume_size[0], feature_volume_size[1],
                          feature_volume_size[2]) )
                         
    # A lookup table for the cache. The filename is used as a key. The
    # values is the index in the cache
    self.cache_entries={}
    # Vice versa: the key for every entry in the cache
    self.key_for_cache_entries=["" for i in range(0,cache_size)]
    self.nextfreeidx=0
    
    # Statistics
    self.no_queries=0
    self.cache_hit=0
                               
  """
    Input: 1x2 numpy array of X and Y coordinate
    Ouput: complete filename for the feature volume
  """
  def coord2filename(self, coord):
    new_x = str('{:+.2f}'.format(coord[0])).zfill(10)
    new_y = str('{:+.2f}'.format(coord[1])).zfill(10)
    file_name = new_x + '_' + new_y
    return file_name

      
  def new_task(self, coord_current_frame, coordinates_nearby_grid):
    #print('New task with current frame coord', coord_current_frame)
    # Number of pairs to infer
    self.n=len(coordinates_nearby_grid)
    # Convert to filenames
    self.map_filenames=[]
    for i in range(0,self.n):
      self.map_filenames.append(self.coord2filename(coordinates_nearby_grid[i]))
      
    self.current_filename=self.coord2filename(coord_current_frame)
    
    # prepare right leg: a repeated version of query
    fcurrent=self.load_feature_volume(self.current_filename)
    self.input1=np.tile(fcurrent,(self.batch_size,1,1,1,))
    
  # Get a feature volume: either from the cache or load it.
  def get_feature_volume(self, batchi):
    #print('get_feature_volume %s' % self.map_filenames[batchi] )
    #sys.stdout.flush()

    self.no_queries+=1
    if self.map_filenames[batchi] in self.cache_entries:
      self.cache_hit+=1
      #print('cache hit')
      #sys.stdout.flush()
    else:
      if len(self.key_for_cache_entries[self.nextfreeidx])>0:
        # cache entry already used, delete from index
        #print('del old entry')
        #sys.stdout.flush()
        del self.cache_entries[self.key_for_cache_entries[self.nextfreeidx]]
        self.key_for_cache_entries[self.nextfreeidx]=''
        
      self.cache[self.nextfreeidx,:,:,:]=self.load_feature_volume(self.map_filenames[batchi])
      
      self.cache_entries[self.map_filenames[batchi]]=self.nextfreeidx
      self.key_for_cache_entries[self.nextfreeidx]=self.map_filenames[batchi]
      self.nextfreeidx+=1
      if self.nextfreeidx==self.cache_size:
        self.nextfreeidx=0

    return self.cache[self.cache_entries[self.map_filenames[batchi]],:,:,:]
    
  def load_feature_volume(self, filename):
    complete_path=self.datasetpath+'/feature_volumes/'+ filename+'.npz'
    #print('load') #print('load %s' % complete_path)
    #sys.stdout.flush()
    if not os.path.exists(complete_path):
      print('ERROR: feature volume %s doest not exist!!!!' %  complete_path)
      return np.zeros((1,360,128))

    return np.load(complete_path)['arr_0']
      

  # implemented interface of Sequence base class
  def __len__(self):
      return int(np.ceil(self.n / float(self.batch_size)))

  # implemented interface of Sequence base class
  def __getitem__(self, idx):
      #print('getitem with ', idx)
      #sys.stdout.flush()

      maxidx=(idx + 1) * self.batch_size

      cb_size= self.batch_size     
      input1=self.input1
      if maxidx>self.n:
          maxidx=self.n
          cb_size= maxidx - idx * self.batch_size
          input1=self.input1[0:cb_size,:,:,:]
          

      input2= np.zeros( (cb_size,  self.feature_volume_size[0], self.feature_volume_size[1],
                          self.feature_volume_size[2]) )   
      d= idx * self.batch_size
      for batchi in range(idx * self.batch_size, maxidx):
        input2[batchi-d,:,:,:]=self.get_feature_volume(batchi)
        
      return ( [input1,input2], 0 )

  def print_statistics(self):
    print('Feature volume cache hit rate: %5.1f %%' % 
      (100.0*self.cache_hit/self.no_queries) )
    
