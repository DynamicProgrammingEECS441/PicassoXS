
import tensorflow as tf

from tensorflow.compat.v1 import get_variable, get_variable_scope, variable_scope


def instance_norm(input, name="instance_norm", is_training=True):
    with tf.compat.v1.variable_scope(name):
        depth = input.get_shape()[3]
        scale = get_variable("scale", [depth], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        offset = get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with variable_scope(name):
        dim = [[4, 4, input_.shape[3], output_dim]]
        filter = tf.Variable(tf.random.normal([ks, ks, input_.shape[3], output_dim], dtype=tf.float32))
        # filter = tf.compat.v1.truncated_normal_initializer(stddev=stddev)
        return tf.nn.conv2d(input_, filter, strides=[1, s, s, 1], padding=padding)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    # Upsampling procedure, like suggested in this article:
    # https://distill.pub/2016/deconv-checkerboard/. At first upsample
    # tensor like an image and then apply convolutions.
    with variable_scope(name):
        input_ = tf.image.resize(images=input_,
                                size=tf.shape(input_)[1:3] * s,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')