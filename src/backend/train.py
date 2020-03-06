import tensorflow as tf
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import multiprocessing

import imageio
from PIL import Image

from module import *
from tensorflow.compat.v1 import placeholder


class Model(object):
    def __init__(self, sess, args):
        self.model_name = args.model_name
        self.root_dir = './models'
        self.checkpoint_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint')
        self.checkpoint_long_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint_long')
        self.sample_dir = os.path.join(self.root_dir, self.model_name, 'sample')
        self.inference_dir = os.path.join(self.root_dir, self.model_name, 'inference')
        self.logs_dir = os.path.join(self.root_dir, self.model_name, 'logs')

        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.loss = sce_criterion
        self.initial_step = 0

        OPTIONS = namedtuple('OPTIONS',
                             'batch_size image_size \
                              total_steps save_freq lr\
                              gf_dim df_dim \
                              is_training \
                              path_to_content_dataset \
                              path_to_art_dataset \
                              discr_loss_weight transformer_loss_weight feature_loss_weight')
        self.options = OPTIONS._make((args.batch_size, args.image_size,
                                      args.total_steps, args.save_freq, args.lr,
                                      args.ngf, args.ndf,
                                      args.phase == 'train',
                                      args.path_to_content_dataset,
                                      args.path_to_art_dataset,
                                      args.discr_loss_weight, args.transformer_loss_weight, args.feature_loss_weight
                                      ))
        # Create all the folders for saving the model
        # if not os.path.exists(self.root_dir):
        #     os.makedirs(self.root_dir)
        # if not os.path.exists(os.path.join(self.root_dir, self.model_name)):
        #     os.makedirs(os.path.join(self.root_dir, self.model_name))
        # if not os.path.exists(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir)
        # if not os.path.exists(self.checkpoint_long_dir):
        #     os.makedirs(self.checkpoint_long_dir)
        # if not os.path.exists(self.sample_dir):
        #     os.makedirs(self.sample_dir)
        # if not os.path.exists(self.inference_dir):
        #     os.makedirs(self.inference_dir)

        self.build()
        # self.saver = tf.compat.v1.train.Saver(max_to_keep=2)
        # self.saver_long = tf.compat.v1.train.Saver(max_to_keep=None)

    def build(self):
        print('build')
        # x = [[5 for i in range(512)] for j in range(512)]
        if self.options.is_training:
            # ==================== Define placeholders. ===================== #
            with tf.name_scope('placeholder'):
                self.input_painting = placeholder(dtype=tf.float32,
                                                     shape=[self.batch_size, None, None, 3],
                                                     name='painting')
                self.input_photo = placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size, None, None, 3],
                                                  name='photo')
                self.lr = placeholder(dtype=tf.float32, shape=(), name='learning_rate')

            # ===================== Wire the graph. ========================= #
            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                options=self.options,
                                                reuse=False)

            # Decode obtained features
            self.output_photo = decoder(features=self.input_photo_features,
                                        options=self.options,
                                        reuse=False)
            
            # Get features of output images. Need them to compute feature loss.
            self.output_photo_features = encoder(image=self.output_photo,
                                                 options=self.options,
                                                 reuse=True)
    
    def train(self, args, ckpt_nmbr=None):
        print('train')