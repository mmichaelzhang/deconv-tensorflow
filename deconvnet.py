from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def deconvnet_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      padding='SAME'):
    with tf.layers.arg_scope([tf.layers.conv2d_transpose],
                             padding = 'SAME', 
                             activation = tf.nn.relu) as arg_sc:
      return arg_sc


def deconvnet(input1,
              input2,
              batch_size
              scope='deconvnet'):
  
  with tf.variable_scope(scope, 'deconvnet', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(input1, 256, [3, 3], stride = 2, scope = 'conv1')
      net = slim.conv2d(net, 512, [3, 3], stride = 2, scope = 'conv2')
      net = slim.conv2d(net, 1024, [3, 3], stride = 2, scope = 'conv3')

      net_2 = slim.conv2d(input2, 512, [3, 3], stride = 2, scope = 'conv1_2')
      net_2 = slim.conv2d(net_2, 1024, [3, 3], stride = 2, scope = 'conv2_2')

      net = tf.concat([net, net_2], 3)

      net = slim.conv2d(net, 1024, [3, 3], stride = 1, scope = 'conv4')
      net = slim.conv2d(net, 512, [3, 3], stride = 1, scope = 'conv5')

      net = tf.layers.conv2d_transpose(net, 256, [4, 4], strides = (2, 2), name = 'deconv6')
      net = tf.layers.conv2d_transpose(net, 128, [4, 4], strides = (2, 2), name = 'deconv7')
      net = tf.layers.conv2d_transpose(net, 64, [4, 4], strides = (2, 2), name = 'deconv8')
      net = tf.layers.conv2d_transpose(net, 32, [4, 4], strides = (2, 2), name = 'deconv9')
      net = tf.layers.conv2d_transpose(net, 3, [4, 4], strides = (2, 2), name = 'deconv10', activation = None)

      
      net = slim.conv2d(net, )

      net = slim.conv2d(inputs, 256, [3, 3], 4, scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      end_points[sc.name + '/deconv10'] = net
      return net, end_points