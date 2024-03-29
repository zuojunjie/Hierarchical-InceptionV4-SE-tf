from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

# from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'dietlens_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 799

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [256 x 256 x 3] color image.',
    'label': 'A single integer between 0 and 799',
}
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if not reader:
    reader = tf.TFRecordReader

  keys_to_features = {
      'train_img': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpg'),
      'train_label': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=-1),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('train_img',shape=[256, 256, 3]),
      'label': slim.tfexample_decoder.Tensor('train_label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  # if dataset_utils.has_labels(dataset_dir):
  #   labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)