import gc
import os
import time
import pandas as pd

from keras.utils import generic_utils
from tensorflow.keras.optimizers import Adam
import numpy as np
import models
import data_utils
import physics
import os
import tensorflow as tf
import h5py

# 获取当前文件所在目录的绝对路径
dir_path = os.path.dirname(os.path.realpath(__file__))




# 创建生成器、判别器和GAN模型，并编译它们
def create_models(scene_size, modis_var_dim, lr_disc, lr_gan):
    # Create optimizers
    opt_disc = Adam(lr_disc, 0.5)
    opt_gan = Adam(lr_gan, 0.5)

    # 创建生成器和判别器模型
    gen = models.cs_generator(scene_size, modis_var_dim)
    disc = models.discriminator(scene_size, modis_var_dim)

    # # 编译GAN和判别器
    disc.trainable = False

    gan = models.cs_modis_cgan(gen, disc, scene_size, modis_var_dim)
    gan.compile(loss='binary_crossentropy', optimizer=opt_gan)
    disc.trainable = True
    disc.compile(loss='binary_crossentropy', optimizer=opt_disc)

    # phy = models.rq(scene_size, modis_var_dim)

    return (gen, disc, gan, opt_disc, opt_gan)







