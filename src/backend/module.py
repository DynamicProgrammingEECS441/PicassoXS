from __future__ import division
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.compat.v1 import get_variable, get_variable_scope, variable_scope
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation

from layers import *


def encoder(image, options, reuse=True, name="encoder"):
    """
    Args:
        image: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new encoder or use existing
        name: name of the encoder

    Returns: Encoded image.
    """

    with variable_scope(name):
        if reuse:
            get_variable_scope().reuse_variables()
        else:
            assert get_variable_scope().reuse is False
        
        def encoder_conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", ins_norm_name="instance_norm", is_training=True):
            return tf.nn.relu(instance_norm(input=conv2d(input_, output_dim, ks, s, stddev, padding, name),
                                            is_training=is_training,
                                            name=ins_norm_name))
        
        image = instance_norm(input=image,
                              is_training=options.is_training,
                              name='g_e0_bn')
        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
        c1 = encoder_conv2d(c0, options.gf_dim, 3, 1, padding='VALID', name='g_e1_c', is_training=options.is_training, ins_norm_name='g_e1_bn')
        c2 = encoder_conv2d(c1, options.gf_dim, 3, 2, padding='VALID', name='g_e2_c', is_training=options.is_training, ins_norm_name='g_e2_bn')
        c3 = encoder_conv2d(c2, options.gf_dim * 2, 3, 2, padding='VALID', name='g_e3_c', is_training=options.is_training, ins_norm_name='g_e3_bn')
        c4 = encoder_conv2d(c3, options.gf_dim * 4, 3, 2, padding='VALID', name='g_e4_c', is_training=options.is_training, ins_norm_name='g_e4_bn')
        c5 = encoder_conv2d(c4, options.gf_dim * 8, 3, 2, padding='VALID', name='g_e5_c', is_training=options.is_training, ins_norm_name='g_e5_bn')
        print('encoder')
        return c5


def decoder(features, options, reuse=True, name="decoder"):
    """
    Args:
        features: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new decoder or use existing
        name: name of the decoder

    Returns: Decoded image.
    """

    with variable_scope(name):
        if reuse:
            get_variable_scope().reuse_variables()
        else:
            assert get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x
        
        # Now stack 9 residual blocks
        num_kernels = features.get_shape().as_list()[-1]
        r1 = residule_block(features, num_kernels, name='g_r1')
        r2 = residule_block(r1, num_kernels, name='g_r2')
        r3 = residule_block(r2, num_kernels, name='g_r3')
        r4 = residule_block(r3, num_kernels, name='g_r4')
        r5 = residule_block(r4, num_kernels, name='g_r5')
        r6 = residule_block(r5, num_kernels, name='g_r6')
        r7 = residule_block(r6, num_kernels, name='g_r7')
        r8 = residule_block(r7, num_kernels, name='g_r8')
        r9 = residule_block(r8, num_kernels, name='g_r9')

        # Decode image.
        def decode_deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d", ins_norm_name="instance_norm", is_training=True):
            return tf.nn.relu(instance_norm(input=deconv2d(input_, output_dim, ks, s, stddev, name),
                                            is_training=is_training,
                                            name=ins_norm_name))

        d1 = decode_deconv2d(r9, options.gf_dim * 8, 3, 2, name='g_d1_dc', ins_norm_name='g_d1_bn', is_training=options.is_training)
        d2 = decode_deconv2d(d1, options.gf_dim * 4, 3, 2, name='g_d2_dc', ins_norm_name='g_d2_bn', is_training=options.is_training)
        d3 = decode_deconv2d(d2, options.gf_dim * 2, 3, 2, name='g_d3_dc', ins_norm_name='g_d3_bn', is_training=options.is_training)
        d4 = decode_deconv2d(d3, options.gf_dim, 3, 2, name='g_d4_dc', ins_norm_name='g_d4_bn', is_training=options.is_training)


        d5 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c'))*2. - 1.
        print('decode')
        return pred


# ====== Define different types of losses applied to discriminator's output. ====== #

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))