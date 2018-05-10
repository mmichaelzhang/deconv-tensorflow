from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from deconvnet import *

# Data sets
DECONV_TRAINING = "deconv_training.csv"
DECONV_TEST = "deconv_test.csv"
LEARNING_RATE = 0.1

def deconvnet_fn(features, labels, mode, params)
  recover_image = deconvnet(features['x1'], features['x2'])
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"image": recover_image})
  loss = tf.losses.mean_squared_error(labels, recover_image)
  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op)

def main():
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DECONV_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=DECONV_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  model_params = {"learning_rate": LEARNING_RATE}

  visulizer = tf.estimator.Estimator(model_fn = deconvnet_fn, params = model_params)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x1={"x1": np.array(training_set.feature1)},
      x2={"x2": np.array(training_set.feature2)},
      y=np.array(training_set.image),
      num_epochs=None,
      shuffle=True)

  classifier.train(input_fn=train_input_fn, steps=5000)

  