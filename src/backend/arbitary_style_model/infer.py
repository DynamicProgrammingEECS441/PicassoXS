#
#   Copyright © 2020. All rights reserved.
#   python == 3.6 
#   tensorflow == 1.14
#   scipy==1.1.0
#

import tensorflow as tf
from model import Model
import utils

def inference(opt):
    content_img_list = utils.list_images(opt.content_img_dir)
    style_img_list = utils.list_images(opt.style_img_dir)

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content_img = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content_img')
        style_img   = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style_img')

        model = Model(opt.checkpoint_encoder)

        generated_img = model.transform(content_img, style_img)

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, opt.checkpoint_model)

        outputs = []
        for content_img_path in content_img_list:
            content_img = utils.get_images(content_img_path, height=opt.img_size, width=opt.img_size)
            for style_img_path in style_img_list:
                style_img = utils.get_images(style_img_path)

                result = sess.run(generated_img, feed_dict={
                    content_img: content_img, 
                    style_img: style_img
                    })

                outputs.append(result[0])

        utils.save_images(outputs, opt.content_img_dir, opt.style_img_dir, opt.output_dir, suffix=opt.style_weight)

    return outputs


