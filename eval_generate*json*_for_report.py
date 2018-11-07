#opyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

##########################
# Attention Module Flags #
##########################

tf.app.flags.DEFINE_string(
    'attention_module', None,
    'The name of attention module to use.')

FLAGS = tf.app.flags.FLAGS

image_path='/home/jjzuo/SENet-tensorflow-slim/VG750_raisin bread/750_Raisin Bread/000430.jpg'
tfrecord_path='/home/jjzuo/SENet-tensorflow-slim/dietlens/test/test-0011-of-25'
def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    image_jpg = tf.gfile.FastGFile(image_path, 'rb').read()
    image = tf.image.decode_jpeg(image_jpg, channels=3)
    filename_queue = tf.train.string_input_producer([tfrecord_path]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image/class/label': tf.FixedLenFeature([],dtype=tf.int64),
                                        'image/encoded': tf.FixedLenFeature([], tf.string)
                                   })  #取出包含image和label的feature对象
    image_2=tf.decode_raw(features['image/encoded'],tf.uint8)
    image_2 = tf.reshape(image_2, [320, 320, 3])
    label = tf.cast(features['image/class/label'], tf.int64)
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=798,
        is_training=False,
        attention_module=FLAGS.attention_module)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    ####################
    # Define the model #
    ####################
    image=tf.expand_dims(image, 0)
    print(image)
    logits, _ = network_fn(image)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)

    #checkpoint_path='/home/jjzuo/SENet-tensorflow-slim/train_logs/checkpoint'
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path
    #init_fn = slim.assign_from_checkpoint_fn( checkpoint_path,slim.get_model_variables())

    tf.logging.info('Evaluating %s' % checkpoint_path)
    
    # GPU memory dynamic allocation
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)
      #init_fn(sess)
      coord=tf.train.Coordinator()
      threads= tf.train.start_queue_runners(coord=coord)
      acc=0 
      for i in range(1):
        l,result = sess.run([label,predictions])
        #sess.run(tf.local_variables_initializer())
        #l ,image_resl= sess.run([label,image])#在会话中取出image和label
        print(result,l)
        if result==l:
             acc+=1
      #print(l)
      print(acc)
      coord.request_stop()
      coord.join(threads)
if __name__ == '__main__':
  tf.app.run()
