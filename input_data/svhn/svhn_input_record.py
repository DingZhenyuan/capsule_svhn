from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import tensorflow as tf


def _read_and_decode(filename_queue, batch_size, image_dim=32, distort=False, split='train'):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_dim, image_dim, 3])
    image.set_shape([image_dim, image_dim, 3])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255)

    if distort:
        cropped_dim = image_dim - 4
        if split == 'train':
            image = tf.reshape(image, [image_dim, image_dim])
            image = tf.random_crop(image, [cropped_dim, cropped_dim])
            # 0.26179938779 is 15 degress in radians
            image = tf.contrib.image.rotate(image, random.uniform(-0.26179938779,
                                                           0.26179938779))
            image = tf.reshape(image, [cropped_dim, cropped_dim, 3])
            image.set_shape([cropped_dim, cropped_dim, 3])
        else:
            fraction = cropped_dim / image_dim
            image = tf.image.central_crop(image, central_fraction=fraction)
            image.set_shape([cropped_dim, cropped_dim, 3])
        image_dim = cropped_dim

    label = tf.cast(features['label'], tf.int32)
    features = {
        'images': image,
        'labels': tf.one_hot(label, 10),
        'recons_image': image,
        'recons_label': label
    }
    return features, image_dim


# input method
def inputs(data_dir,
           batch_size,
           split,
           distort=False,
           batch_capacity = 10000
           ):

    filename = [os.path.join(data_dir, '{}.tfrecords').format(split)]
    # filename = "svhn_data/format2/train.tfrecords"

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filename, shuffle=False)

        features, image_dim = _read_and_decode(filename_queue, batch_size)

        if split == 'train':
            batched_features = tf.train.shuffle_batch(
                features,
                batch_size=batch_size,
                num_threads=2,
                capacity=70000 + 3 * batch_size,
                min_after_dequeue=70000
            )
        else:
            batched_features = tf.train.batch(
                features,
                batch_size=batch_size,
                num_threads=1,
                capacity=20000 + 3 * batch_size
            )
        batched_features['height'] = image_dim
        batched_features['depth'] = 3
        batched_features['num_targets'] = 1
        batched_features['num_classes'] = 10
        return batched_features
