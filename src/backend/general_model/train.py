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
import cv2

import img_augm
import prepare_dataset

from module import *
from tensorflow.compat.v1 import placeholder, summary, global_variables_initializer

def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    #print("arr shape", arr.shape)
    return arr/127.5 - 1.

def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return ((arr + 1.) * 127.5).astype(np.uint8)

def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save
    Returns:
    """
    def batch_to_img(batch):
        return np.reshape(batch,
                          newshape=(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3]))

    inputs = np.concatenate([batch_to_img(input_painting_batch), batch_to_img(input_photo_batch)],
                            axis=0)
    outputs = np.concatenate([batch_to_img(output_painting_batch), batch_to_img(output_photo_batch)],
                             axis=0)

    to_save = np.concatenate([inputs,outputs], axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)

    # scipy.misc.imsave(filepath, arr=to_save)
    cv2.imwrite(filepath, to_save)
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
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(os.path.join(self.root_dir, self.model_name)):
            os.makedirs(os.path.join(self.root_dir, self.model_name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_long_dir):
            os.makedirs(self.checkpoint_long_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.inference_dir):
            os.makedirs(self.inference_dir)

        self.build()
        self.saver = tf.compat.v1.train.Saver(max_to_keep=2)
        self.saver_long = tf.compat.v1.train.Saver(max_to_keep=None)

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

            # Add discriminators.
            # Note that each of the predictions contain multiple predictions at different scale.
            self.input_painting_discr_predictions = discriminator(image=self.input_painting,
                                                                  options=self.options,
                                                                  reuse=False)
            self.input_photo_discr_predictions = discriminator(image=self.input_photo,
                                                               options=self.options,
                                                               reuse=True)
            self.output_photo_discr_predictions = discriminator(image=self.output_photo,
                                                                options=self.options,
                                                                reuse=True)

            # ===================== Final losses that we optimize. ===================== #

            # Discriminator.
            # Have to predict ones only for original paintings, otherwise predict zero.
            scale_weight = {"scale_0": 1.,
                            "scale_1": 1.,
                            "scale_3": 1.,
                            "scale_5": 1.,
                            "scale_6": 1.}
            self.input_painting_discr_loss = {key: self.loss(pred, tf.ones_like(pred)) * scale_weight[key]
                                              for key, pred in zip(self.input_painting_discr_predictions.keys(),
                                                                   self.input_painting_discr_predictions.values())}
            self.input_photo_discr_loss = {key: self.loss(pred, tf.zeros_like(pred)) * scale_weight[key]
                                           for key, pred in zip(self.input_photo_discr_predictions.keys(),
                                                                self.input_photo_discr_predictions.values())}
            self.output_photo_discr_loss = {key: self.loss(pred, tf.zeros_like(pred)) * scale_weight[key]
                                            for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                 self.output_photo_discr_predictions.values())}

            self.discr_loss = tf.add_n(list(self.input_painting_discr_loss.values())) + \
                              tf.add_n(list(self.input_photo_discr_loss.values())) + \
                              tf.add_n(list(self.output_photo_discr_loss.values()))
            
            # Compute discriminator accuracies.
            self.input_painting_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                                         dtype=tf.float32)) * scale_weight[key]
                                             for key, pred in zip(self.input_painting_discr_predictions.keys(),
                                                                  self.input_painting_discr_predictions.values())}
            self.input_photo_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                                      dtype=tf.float32)) * scale_weight[key]
                                          for key, pred in zip(self.input_photo_discr_predictions.keys(),
                                                               self.input_photo_discr_predictions.values())}
            self.output_photo_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                                       dtype=tf.float32)) * scale_weight[key]
                                           for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                self.output_photo_discr_predictions.values())}
            self.discr_acc = (tf.add_n(list(self.input_painting_discr_acc.values())) + \
                              tf.add_n(list(self.input_photo_discr_acc.values())) + \
                              tf.add_n(list(self.output_photo_discr_acc.values()))) / float(len(scale_weight.keys())*3)

            # Generator.
            # Predicts ones for both output images.
            self.output_photo_gener_loss = {key: self.loss(pred, tf.ones_like(pred)) * scale_weight[key]
                                            for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                 self.output_photo_discr_predictions.values())}

            self.gener_loss = tf.add_n(list(self.output_photo_gener_loss.values()))

            # Compute generator accuracies.
            self.output_photo_gener_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                                       dtype=tf.float32)) * scale_weight[key]
                                           for key, pred in zip(self.output_photo_discr_predictions.keys(),
                                                                self.output_photo_discr_predictions.values())}

            self.gener_acc = tf.add_n(list(self.output_photo_gener_acc.values())) / float(len(scale_weight.keys()))

            # Image loss.
            self.img_loss_photo = mse_criterion(transformer_block(self.output_photo),
                                                transformer_block(self.input_photo))
            self.img_loss = self.img_loss_photo

            # Features loss.
            self.feature_loss_photo = abs_criterion(self.output_photo_features, self.input_photo_features)
            self.feature_loss = self.feature_loss_photo

            # ================== Define optimization steps. =============== #
            t_vars = tf.compat.v1.trainable_variables()
            self.discr_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.encoder_vars = [var for var in t_vars if 'encoder' in var.name]
            self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]

            # Discriminator and generator steps.
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.d_optim_step = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.discr_loss_weight * self.discr_loss,
                    var_list=[self.discr_vars])
                self.g_optim_step = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.discr_loss_weight * self.gener_loss +
                         self.options.transformer_loss_weight * self.img_loss +
                         self.options.feature_loss_weight * self.feature_loss,
                    var_list=[self.encoder_vars + self.decoder_vars])
            
            # ============= Write statistics to tensorboard. ================ #

            # Discriminator loss summary.
            s_d1 = [summary.scalar("discriminator/input_painting_discr_loss/"+key, val)
                    for key, val in zip(self.input_painting_discr_loss.keys(), self.input_painting_discr_loss.values())]
            s_d2 = [summary.scalar("discriminator/input_photo_discr_loss/"+key, val)
                    for key, val in zip(self.input_photo_discr_loss.keys(), self.input_photo_discr_loss.values())]
            s_d3 = [summary.scalar("discriminator/output_photo_discr_loss/" + key, val)
                    for key, val in zip(self.output_photo_discr_loss.keys(), self.output_photo_discr_loss.values())]
            s_d = summary.scalar("discriminator/discr_loss", self.discr_loss)
            self.summary_discriminator_loss = summary.merge(s_d1+s_d2+s_d3+[s_d])

            # Discriminator acc summary.
            s_d1_acc = [summary.scalar("discriminator/input_painting_discr_acc/"+key, val)
                    for key, val in zip(self.input_painting_discr_acc.keys(), self.input_painting_discr_acc.values())]
            s_d2_acc = [summary.scalar("discriminator/input_photo_discr_acc/"+key, val)
                    for key, val in zip(self.input_photo_discr_acc.keys(), self.input_photo_discr_acc.values())]
            s_d3_acc = [summary.scalar("discriminator/output_photo_discr_acc/" + key, val)
                    for key, val in zip(self.output_photo_discr_acc.keys(), self.output_photo_discr_acc.values())]
            s_d_acc = summary.scalar("discriminator/discr_acc", self.discr_acc)
            s_d_acc_g = summary.scalar("discriminator/discr_acc", self.gener_acc)
            self.summary_discriminator_acc = summary.merge(s_d1_acc+s_d2_acc+s_d3_acc+[s_d_acc])

            # Image loss summary.
            s_i1 = summary.scalar("image_loss/photo", self.img_loss_photo)
            s_i = summary.scalar("image_loss/loss", self.img_loss)
            self.summary_image_loss = summary.merge([s_i1 + s_i])

            # Feature loss summary.
            s_f1 = summary.scalar("feature_loss/photo", self.feature_loss_photo)
            s_f = summary.scalar("feature_loss/loss", self.feature_loss)
            self.summary_feature_loss = summary.merge([s_f1 + s_f])

            self.summary_merged_all = summary.merge_all()
            self.writer = summary.FileWriter(self.logs_dir, self.sess.graph)
        else:
            # ==================== Define placeholders. ===================== #
            with tf.name_scope('placeholder'):
                self.input_photo = placeholder(dtype=tf.float32,
                                                  shape=[self.batch_size, None, None, 3],
                                                  name='photo')
        
            # ===================== Wire the graph. ========================= #
            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                options=self.options,
                                                reuse=False)

            # Decode obtained features.
            self.output_photo = decoder(features=self.input_photo_features,
                                        options=self.options,
                                        reuse=False)


    def train(self, args):
        # Initialize augmentor.
        augmentor = img_augm.Augmentor(crop_size=[self.options.image_size, self.options.image_size],
                                       hue_augm_shift=0.05,
                                       saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                                       value_augm_shift=0.05, value_augm_scale=0.05, )

        content_dataset_places = prepare_dataset.PlacesDataset(path_to_dataset=self.options.path_to_content_dataset)
        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset)

        # Now initialize the graph
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        print("Start training.")
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            if self.load(self.checkpoint_long_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Initial discriminator success rate.
        win_rate = args.discr_success_rate
        discr_success = args.discr_success_rate
        alpha = 0.05

        for step in tqdm(range(self.initial_step, self.options.total_steps+1),
                         initial=self.initial_step,
                         total=self.options.total_steps):
            #print('step {}'.format(step))
            batch_art = art_dataset.get_batch(augmentor=augmentor, batch_size=self.batch_size)
            batch_content = content_dataset_places.get_batch(augmentor=augmentor, batch_size=self.batch_size)
            if discr_success >= win_rate:
                # Train generator
                _, summary_all, gener_acc_ = self.sess.run(
                    [self.g_optim_step, self.summary_merged_all, self.gener_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })
                discr_success = discr_success * (1. - alpha) + alpha * (1. - gener_acc_)
            else:
                # Train discriminator.
                _, summary_all, discr_acc_ = self.sess.run(
                    [self.d_optim_step, self.summary_merged_all, self.discr_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })

                discr_success = discr_success * (1. - alpha) + alpha * discr_acc_
            self.writer.add_summary(summary_all, step * self.batch_size)

            if step % self.options.save_freq == 0 and step > self.initial_step:
                self.save(step)

            # And additionally save all checkpoints each 15000 steps.
            if step % 15000 == 0 and step > self.initial_step:
                self.save(step, is_long=True)

            if step % 500 == 0:
                output_paintings_, output_photos_= self.sess.run(
                    [self.input_painting, self.output_photo],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.lr: self.options.lr
                    })

                save_batch(input_painting_batch=batch_art['image'],
                           input_photo_batch=batch_content['image'],
                           output_painting_batch=denormalize_arr_of_imgs(output_paintings_),
                           output_photo_batch=denormalize_arr_of_imgs(output_photos_),
                           filepath='%s/step_%d.jpg' % (self.sample_dir, step))

        print("Done.")
        
    def inference(self, path_to_folder, to_save_dir=None):

        init_op = global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        elif self.load(self.checkpoint_long_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d' % (self.initial_step, self.image_size))

        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for _, img_path in enumerate(tqdm(names)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_shape = img.shape[:2]
        
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)

            img = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img),
                })

            img = img[0]
            img = denormalize_arr_of_imgs(img)
            img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
            img_name = os.path.basename(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(to_save_dir, img_name[:-4] + "_stylized.jpg"), img)
        print("Inference is finished.")

    def save(self, step, is_long=False):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if is_long:
            self.saver_long.save(self.sess,
                                 os.path.join(self.checkpoint_long_dir, self.model_name+'_%d.ckpt' % step),
                                 global_step=step)
        else:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, self.model_name + '_%d.ckpt' % step),
                            global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
        ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
            print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
