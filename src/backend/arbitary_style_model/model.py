#
#   Copyright © 2020. All rights reserved.
#   python >= 3.6 
#   tensorflow >= 1.2 
#

import tensorflow as tf
import numpy as np 

def AdaIN(content, style, epsilon=1e-5):
    meanC, varC = tf.nn.moments(content, [1, 2], keep_dims=True)
    meanS, varS = tf.nn.moments(style,   [1, 2], keep_dims=True)

    sigmaC = tf.sqrt(tf.add(varC, epsilon))
    sigmaS = tf.sqrt(tf.add(varS, epsilon))
    
    return (content - meanC) * sigmaS / sigmaC + meanS


class Decoder(object):

    def __init__(self):
        self.weight_vars = []

        with tf.variable_scope('decoder'):
            self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
            self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

            self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(128,  64, 3, scope='conv2_1'))

            self.weight_vars.append(self._create_variables( 64,  64, 3, scope='conv1_2'))
            self.weight_vars.append(self._create_variables( 64,   3, 3, scope='conv1_1'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape  = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape, name='kernel')
            bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=[output_filters], name='bias')
            return (kernel, bias)

    def decode(self, image):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 6)
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in upsample_indices:
                out = upsample(out)

        return out


def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    if use_relu:
        out = tf.nn.relu(out)

    return out


def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width  = tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output



ENCODER_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1'
)


class Encoder(object):

    def __init__(self, weights_path):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)

        idx = 0
        self.weight_vars = []

        # create the TensorFlow variables
        with tf.variable_scope('encoder'):
            for layer in ENCODER_LAYERS:
                kind = layer[:4]

                if kind == 'conv':
                    kernel = weights['arr_%d' % idx].transpose([2, 3, 1, 0])
                    bias   = weights['arr_%d' % (idx + 1)]
                    kernel = kernel.astype(np.float32)
                    bias   = bias.astype(np.float32)
                    idx += 2

                    with tf.variable_scope(layer):
                        W = tf.Variable(kernel, trainable=False, name='kernel')
                        b = tf.Variable(bias,   trainable=False, name='bias')

                    self.weight_vars.append((W, b))

    def encode(self, image):
        # create the computational graph
        idx = 0
        layers = {}
        current = image

        for layer in ENCODER_LAYERS:
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel, bias)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current

        assert(len(layers) == len(ENCODER_LAYERS))

        enc = layers[ENCODER_LAYERS[-1]]

        return enc, layers

    def preprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])

    def deprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image + np.array([103.939, 116.779, 123.68])
        else:
            return image + np.array([123.68, 116.779, 103.939])


def conv2d(x, kernel, bias):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Model(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def transform(self, content, style):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        style   = tf.reverse(style,   axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)
        style   = self.encoder.preprocess(style)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers

        # pass the encoded images to AdaIN
        target_features = AdaIN(enc_c, enc_s)
        self.target_features = target_features

        # decode target features back to image
        generated_img = self.decoder.decode(target_features)

        # deprocess image
        generated_img = self.encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

