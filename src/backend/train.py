import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy

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


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
    
    def train(self, args, ckpt_nmbr=None):
        pass