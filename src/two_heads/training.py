#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Laebe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script is used to train a network which can predict overlap and orientation simultaneously

"""
Train for Overlap network.

Commandline arguments:
  If no arguments are given, the configuration file 'network.yml' in
  the current directory is used.

  You can also execute this script with

  training.py <yml-config-file>

  Then the argument should be the yaml configuration file which you intend
  to use.
"""
import datetime
# To get a log file
import importlib
import logging
import os
import sys

import keras.optimizers
import numpy as np
import tensorflow as tf
import yaml
from keras import backend as K
from keras.callbacks import LearningRateScheduler

importlib.reload(logging)  # needed for ipython console

import generateNet
from ImagePairOverlapOrientationSequence import ImagePairOverlapOrientationSequence
from overlap_orientation_npz_file2string_string_nparray import overlap_orientation_npz_file2string_string_nparray


# ============ file global variables (used in functions) ======================
network_output_size = 0 
min_overlap_for_angle = 0.7


# ============ local functions (loss, learning rate) ==========================
def learning_rate_schedule(initial_lr=1e-3, alpha=0.99):
  """Wrapper function to create a LearningRateScheduler with step decay schedule.
  """
  
  def scheduler(epoch):
    if epoch < 1:
      return initial_lr * 0.1
    else:
      return initial_lr * np.power(alpha, epoch - 1.0)

  return LearningRateScheduler(scheduler)


class LossHistory(keras.callbacks.Callback):
  """ Small class for callback to record loss after each batch.
  """
  
  def on_train_begin(self, logs={}):
    self.losses = []
  
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))


def my_sigmoid_loss(y_true, y_pred):
  """ Loss function in form of a mean sigmoid. Used for overlap.
      This is an alternative for mean squard error where
      - the loss for small differences is smaller than squared diff
      - the loss for large errors is kind of equal

   In Matlab:   1./(1+exp(-((diff+0.25)*24-12))), diff is absolute difference
  """
  diff = K.abs(y_pred - y_true)
  sigmoidx = (diff + 0.25) * 24 - 12
  loss = K.mean(1 / (1 + K.exp(-sigmoidx)))
  
  return loss

  
def my_entropy(y_true, y_pred):
  """ Cross entropy loss for yaw angle prediction.
      Uses the global variables network_output_size and min_overlap_for_angle.
  """      
  y_true = K.greater(y_true, min_overlap_for_angle)
  y_true = K.cast(y_true, dtype='float32')
  return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, network_output_size, name='loss')


# ==================== main script ============================================
# Settings (mostly from yaml file)
# --------------------------------
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

configfilename = 'network.yml'
if len(sys.argv) > 1:
  configfilename = sys.argv[1]

config = yaml.load(open(configfilename))
logger.info("Using configuration file %s." % configfilename)

# overlaps npz files root path
if 'data_root_folder' in config:
  data_root_folder = config['data_root_folder']
else:
  data_root_folder = ''

# Image Path: Use path from yml or data_root_folder if not given
if 'imgpath' in config:
  imgpath = config['imgpath']
else:
  imgpath = data_root_folder

# Training sequences: If given, in every sequence directory, there should be
# the files ground_truth/train_set.npz and ground_truth/validation_set.npz
# If not given, single npz files given by 'traindata_npzfile' and 'validationdata_npzfile' are assumed
if 'training_seqs' in config:
  logger.info('Using multiple npz files for train/validation data ...')
  training_seqs = config['training_seqs']
  training_seqs = training_seqs.split()
  
  traindata_npzfiles = [os.path.join(data_root_folder, seq, 'ground_truth/train_set.npz') for seq in training_seqs]
  validationdata_npzfiles = [os.path.join(data_root_folder, seq, 'ground_truth/validation_set.npz') for seq in training_seqs]
else:
  logger.info('Using a single npz file for train/validation data ...')
  traindata_npzfiles = [config['traindata_npzfile']]
  validationdata_npzfiles = [config['validationdata_npzfile']]

# no channels for input
if 'use_depth' in config:
  use_depth = config['use_depth']
else:
  use_depth = True

if 'use_normals' in config:
  use_normals = config['use_normals']
else:
  use_normals = True

if 'use_class_probabilities' in config:
  use_class_probabilities = config['use_class_probabilities']
else:
  use_class_probabilities = False

if 'use_class_probabilities_pca' in config:
    use_class_probabilities_pca=config['use_class_probabilities_pca']
else:
    use_class_probabilities_pca=False
  
if 'use_intensity' in config:
  use_intensity = config['use_intensity']
else:
  use_intensity = False

no_input_channels = 0
if use_depth:
  no_input_channels += 1
  
if use_normals:
  no_input_channels += 3
  
if use_class_probabilities:
    if use_class_probabilities_pca:
        no_input_channels+=3
    else:
        no_input_channels+=20
      
if use_intensity:
  no_input_channels += 1

if 'rotate_training_data' in config:
  rotate_training_data = config['rotate_training_data']
else:
  rotate_training_data = 0

# Name of model
modelType = config['model']['modelType']
# Input shape of model
inputShape = config['model']['inputShape']
if len(inputShape) == 2:
  inputShape.append(no_input_channels)
else:
  inputShape[2] = no_input_channels

# weights from older training. Not used if empty
pretrained_weightsfilename = config['pretrained_weightsfilename']

# Path where all experiments are stored. Data of current experiment
# will be in experiment_path/testname
experiments_path = config['experiments_path']

# String which defines experiment
testname = config['testname']
os.makedirs(os.path.join(experiments_path, testname), exist_ok=True)

# Logging should go to experiment directory
fileHandler = logging.FileHandler("{0}/training.log".format(os.path.join(experiments_path, testname)),
                                  mode='w')
fileHandler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s",
                                           datefmt='%H:%M:%S'))
logger.addHandler(fileHandler)

# weights filename: if empty, not saved
weights_filename = os.path.join(os.path.join(experiments_path, testname),
                                modelType + '_' + testname + '.weight')

# learning_rate 
initial_lr = config['learning_rate']
# learning rate decay with alpha^ (epoch)
if 'lr_alpha' in config:
    lr_alpha = config['lr_alpha']
else:
    lr_alpha = 0.99
momentum = config['momentum']
batch_size = config['batch_size']
no_batches_in_epoch = config['no_batches_in_epoch']
no_epochs = config['no_epochs']
no_test_pairs = config['no_test_pairs']
# minimal overlap where an orientation angle could be computed
if 'min_overlap_for_angle' in config:
    min_overlap_for_angle = config['min_overlap_for_angle']
else:
    min_overlap_for_angle = 0.7

# Tensorflow log dir
time = datetime.datetime.now().time()
log_dir = "%s/%s/tblog/%s_%s_%d_%d" % (experiments_path, testname, modelType, testname, time.hour, time.minute)
if not os.path.exists(log_dir):
  os.makedirs(log_dir)


# Create network
network_generate_method = getattr(generateNet, 'generate' + modelType)
model = network_generate_method(inputShape, config['model'], smallNet=False)

network_output_size = model.get_layer('orientation_output').output_shape[1]

logger.info("Created neural net %s with %d parameters." %
            (modelType, model.count_params()))
if 'legsType' in config['model']:
  logger.info("  Used net for legs: %s" % config['model']['legsType'])
if 'headType' in config['model']:
  logger.info("  Used net for head: %s" % config['model']['headType'])

learning_rate = learning_rate_schedule(initial_lr=initial_lr, alpha=lr_alpha)
optimizer = keras.optimizers.Adagrad(lr=initial_lr)

losses = {"overlap_output": my_sigmoid_loss, "orientation_output": my_entropy}

lossWeights = {"overlap_output": 5.0, "orientation_output": 1.0}

model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

logger.info("compiled model with learning_rate=%f, lr_alpha=%f, momentum=%f" %
            (initial_lr, lr_alpha, momentum))

if len(pretrained_weightsfilename) > 0:
  logger.info("Load old weights from %s" % pretrained_weightsfilename)
  model.load_weights(pretrained_weightsfilename)
  
print(model.summary())

# Prepare loading of data for training
logger.info("load training data ...")
(train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, train_orientation) = \
  overlap_orientation_npz_file2string_string_nparray(traindata_npzfiles)

n = train_overlap.size
if batch_size * no_batches_in_epoch < n:
  # Make training set smaller
  n = batch_size * no_batches_in_epoch
  train_imgf1 = train_imgf1[0:n]
  train_imgf2 = train_imgf2[0:n]
  train_dir1 = train_dir1[0:n]
  train_dir2 = train_dir2[0:n]
  train_overlap = train_overlap[0:n]
  train_orientation = train_orientation[0:n]
else:
  no_batches_in_epoch = int(np.floor(n / float(batch_size)))

logger.info("load validation data ...")
(validation_imgf1, validation_imgf2, validation_dir1, validation_dir2, validation_overlap, validation_orientation) = \
  overlap_orientation_npz_file2string_string_nparray(validationdata_npzfiles, shuffle=False)
if no_test_pairs < validation_overlap.size:
  # Make test set smaller
  validation_imgf1 = validation_imgf1[0:no_test_pairs]
  validation_imgf2 = validation_imgf2[0:no_test_pairs]
  validation_dir1 = validation_dir1[0:no_test_pairs]
  validation_dir2 = validation_dir2[0:no_test_pairs]
  validation_overlap = validation_overlap[0:no_test_pairs]
  validation_orientation = validation_orientation[0:no_test_pairs]
else:
  no_test_pairs = validation_overlap.size


# Training loop
logger.info("Training loop, saving weights to %s" % weights_filename)
logger.info("  batch size is           : %d" % batch_size)
logger.info("  number of training pairs: %d" % n)
logger.info("  number of test pairs    : %d" % no_test_pairs)
if rotate_training_data == 0:
  logger.info("  NO rotation of training data")
if rotate_training_data == 1:
  logger.info("  rotation of training data, same sequence for every epoch")
if rotate_training_data == 2:
  logger.info("  rotation of training data, different sequence for every epoch")

train_generator = ImagePairOverlapOrientationSequence(imgpath,
                                                      train_imgf1, train_imgf2, train_dir1, train_dir2,
                                                      train_overlap, train_orientation, network_output_size,
                                                      batch_size, inputShape[0], inputShape[1], no_input_channels,
                                                      use_depth=use_depth,
                                                      use_normals=use_normals,
                                                      use_class_probabilities=use_class_probabilities,
                                                      use_intensity=use_intensity,
                                                      use_class_probabilities_pca=use_class_probabilities_pca,
                                                      rotate_data=rotate_training_data)
validation_generator = ImagePairOverlapOrientationSequence(imgpath,
                                          validation_imgf1, validation_imgf2, validation_dir1, validation_dir2, validation_overlap,
                                          validation_orientation, network_output_size,
                                          batch_size, inputShape[0], inputShape[1], no_input_channels,
                                          use_depth=use_depth,
                                          use_normals=use_normals,
                                          use_class_probabilities=use_class_probabilities,
                                          use_intensity=use_intensity,
                                          use_class_probabilities_pca=use_class_probabilities_pca)

batchLossHistory = LossHistory()
for epoch in range(0, no_epochs):
  history = model.fit_generator(train_generator,
                                initial_epoch=epoch, epochs=epoch + 1,
                                callbacks=[batchLossHistory, learning_rate],
                                max_queue_size=10, workers=4,
                                use_multiprocessing=False
                                )
  epoch_loss = history.history['loss'][0]
  learning_rate_hist = K.eval(model.optimizer.lr)
  
  # Saving weights
  if len(weights_filename) > 0:
    logger.info("                  saving model weights ...")
    model.save(weights_filename)
  
  # Evaluation on test data
  logger.info("  Evaluation on test data ...")
  model_outputs = model.predict_generator(validation_generator, max_queue_size=10,
                                          workers=4, use_multiprocessing=False, verbose=1)

  # Statistics for tensorboard logging: plots can be grouped using path notation !
  writer = tf.summary.FileWriter(log_dir)
  losstag0 = "Training/epoch loss"
  losstag1 = "Training/training loss"
  losstag2 = "Training/learning rate"

  losstag14 = "Validation overlap/Max error"
  losstag15 = "Validation overlap/RMS error"

  # metrics for overlap estimation
  diffs = abs(np.squeeze(model_outputs[0])-validation_overlap)
  mean_diff = np.mean(diffs)
  mean_square_error = np.mean(diffs*diffs)
  rms_error = np.sqrt(mean_square_error)
  max_error = np.max(diffs)
  logger.info("  Evaluation on test data results: ")  
  logger.info("           Evaluation: mean overlap difference:   %f" % mean_diff)
  logger.info("           Evaluation: max  overlap difference:   %f" % max_error)
  logger.info("           Evaluation: RMS  overlap error        : %f" % rms_error)   

  summary = tf.Summary(value=[tf.Summary.Value(tag=losstag14,
                                               simple_value=max_error)])
  writer.add_summary(summary, epoch) 

  summary = tf.Summary(value=[tf.Summary.Value(tag=losstag15,
                                               simple_value=rms_error)])
  writer.add_summary(summary, epoch)
  
  # orientation RMS for different overlap thresholds
  summary = tf.Summary(value=[tf.Summary.Value(tag=losstag0,
                                               simple_value=epoch_loss)])
  writer.add_summary(summary, epoch)
  
  for j in range(0, len(batchLossHistory.losses)):
    summary = tf.Summary(value=[tf.Summary.Value(tag=losstag1,
                                                 simple_value=batchLossHistory.losses[j])])
    writer.add_summary(summary, j + epoch * no_batches_in_epoch)
  
  summary = tf.Summary(value=[tf.Summary.Value(tag=losstag2,
                                               simple_value=learning_rate_hist)])
  writer.add_summary(summary, epoch)

  overlap_thres = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  
  for overlap_thre in overlap_thres:
    diffs = []
    for idx in range(len(model_outputs[1])):
      angle = np.argmax(model_outputs[1][idx])
      overlap = np.max(model_outputs[0][idx])
      if overlap > overlap_thre:
        diffs.append(min(abs(angle - validation_orientation[idx]),
                   network_output_size - abs(angle - validation_orientation[idx])))
    diffs = np.array(diffs)
    if diffs.any():
      max_error = np.max(diffs)
      mean_square_error = np.mean(diffs * diffs)
      rms_error = np.sqrt(mean_square_error)
      summary = tf.Summary(value=[tf.Summary.Value(tag='orientation RMS different overlap thresholds/'+str(overlap_thre),
                                                   simple_value=rms_error)])
      writer.add_summary(summary, epoch)
  
  writer.flush()
  
  logger.info("iteration {}, batch/epoch loss: {:.9f}  /  {:.9f}"
              .format(epoch + 1, batchLossHistory.losses[-1], epoch_loss))
