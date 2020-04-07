#
#   Copyright © 2020. All rights reserved.
#   python >= 3.6 
#   tensorflow >= 1.2 
#

import numpy as np
import tensorflow as tf
import utils 
from model import Model

def train(opt):
    '''
    Input:
        opt : optins for training model
    '''
    content_img_list = utils.list_images(opt.content_img_dir)
    style_img_list = utils.list_images(opt.style_img_dir)
    
    with tf.Graph().as_default(), tf.Session() as sess:
        content_img = tf.placeholder(tf.float32, shape=(opt.batch_size, opt.img_size, opt.img_size, 3), name='content_img')
        style_img   = tf.placeholder(tf.float32, shape=(opt.batch_size, opt.img_size, opt.img_size, 3), name='style_img')

        model = Model(opt.checkpoint_encoder)
        
        # Encode Image 
        generated_img = model.transform(content_img, style_img)
        generated_img = tf.reverse(generated_img, axis=[-1])  
        generated_img = model.encoder.preprocess(generated_img)  
        enc_gen, enc_gen_layers = model.encoder.encode(generated_img)

        target_features = model.target_features

        # Content Loss 
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(enc_gen - target_features), axis=[1, 2]))

        # Style Loss 
        style_layer_loss = []
        style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        for layer in style_layers:
            enc_style_feat = model.encoded_style_layers[layer]
            enc_gen_feat   = enc_gen_layers[layer]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + opt.epsilon)
            sigmaG = tf.sqrt(varG + opt.epsilon)

            l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # Total loss 
        loss = opt.content_weight * content_loss + opt.style_weight * style_loss

        # Train 
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(opt.lr, global_step, opt.lr_decay_step, opt.lr_decay)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        # saver
        saver = tf.train.Saver(max_to_keep=10)

        step = 0
        n_batches = int(len(content_img_list) // opt.batch_size)

        try:
            for epoch in range(opt.epoch):

                np.random.shuffle(content_img_list)
                np.random.shuffle(style_img_list)

                for batch in range(n_batches):
                    # retrive a batch of content and style images
                    content_batch_path = content_img_list[batch*opt.batch_size:(batch*opt.batch_size + opt.batch_size)]
                    style_batch_path   = style_img_list[batch*opt.batch_size:(batch*opt.batch_size + opt.batch_size)]

                    content_batch = utils.get_train_images(content_batch_path, crop_height=opt.image.size, crop_width=opt.image.size)
                    style_batch   = utils.get_train_images(style_batch_path,   crop_height=opt.image.size, crop_width=opt.image.size)

                    # run the training step
                    sess.run(train_op, feed_dict={
                        content_img: content_batch, 
                        style_img: style_batch
                        })

                    step += 1

                    if step % 1000 == 0:
                        saver.save(sess, opt.checkpoint_save_dir, global_step=step, write_meta_graph=False)

        except Exception as ex:
            saver.save(sess, opt.checkpoint_save_dir, global_step=step)
            print('Error message: %s' % str(ex))

        # Finish 
        saver.save(sess, opt.checkpoint_save_dir)



