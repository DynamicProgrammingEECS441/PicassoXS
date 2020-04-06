#
#   Copyright © 2020. All rights reserved.
#   python >= 3.6 
#   tensorflow >= 1.2 
#

import os 
import numpy as np 
import tensorflow as tf 
from PIL import Image 
from matplotlib import pyplot as plt
print('tensorflow version', tf.__version__)

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
import imageio 

from style_transfer_net import StyleTransferNet
from utils import get_images, save_images

OUTPUTS_DIR = 'outputs'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
MODEL_SAVE_PATH = 'models/style_weight_2e0.ckpt'

content_img = imageio.imread('./images/content/karya.jpg')
content_img = np.expand_dims(content_img, axis=0)
style_img = imageio.imread('./images/style/mosaic.jpg')
style_img = np.expand_dims(style_img, axis=0)

sess = tf.InteractiveSession()

content = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content')
style   = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style')

model = StyleTransferNet(ENCODER_WEIGHTS_PATH)
output = model.transform(content, style)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, MODEL_SAVE_PATH)

output_img = sess.run(output, feed_dict={content: content_img, style: style_img})

output_img = output_img[0]
print('output_img shape', output_img.shape)
print('output img type', type(output_img))
print('output img dtype', output_img.dtype)
print('output img range max : {}, min : {} '.format(np.max(output_img), np.min(output_img)))

#plt.figure()
#plt.axis('off')
#plt.imshow(output_img)
#plt.show()
output_img_pil = Image.fromarray(np.uint8(output_img))
output_img_pil.save('test.png')


MODEL_NAME = 'arbitary_style'

tensor_info_content = utils.build_tensor_info(content)
tensor_info_style = utils.build_tensor_info(style)
tensor_info_output = utils.build_tensor_info(output)

generate_signature = signature_def_utils.build_signature_def(
      inputs={'content_img': tensor_info_content,
              'style_img':tensor_info_style},
      outputs={'output_img': tensor_info_output},
      method_name=signature_constants.PREDICT_METHOD_NAME)

# Save model 
export_style_transfer_path = './servable/{}/1'.format(MODEL_NAME)
builder = saved_model_builder.SavedModelBuilder(export_style_transfer_path)
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    signature_def_map={'predict_images':generate_signature,}
    )
builder.save()


