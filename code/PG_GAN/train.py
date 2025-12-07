import gc
import os
import time
import pandas as pd

from keras.utils import generic_utils
from tensorflow.keras.optimizers import Adam
import numpy as np
import models
import data_utils
# import physics
import os
import tensorflow as tf
import h5py

# 获取当前文件所在目录的绝对路径
dir_path = os.path.dirname(os.path.realpath(__file__))


# 返回指定模型名称和 epoch 训练轮次的模型权重文件路径
def model_state_paths(model_name, epoch, model_dir=None):
    if model_dir is None:
        model_dir = dir_path + "/../models/%s/" % model_name
    # 构建并返回一个字典，包含生成器、判别器、优化器权重的路径
    paths = {
        "gen_weights_path": os.path.join(
            model_dir + "/gen_weights_epoch%s.h5" % epoch),
        "disc_weights_path": os.path.join(
            model_dir + "/disc_weights_epoch%s.h5" % epoch),
        "opt_disc_weights_path": os.path.join(
            model_dir + "/opt_disc_weights_epoch%s.h5" % epoch),
        "opt_gan_weights_path": os.path.join(
            model_dir + "/opt_gan_weights_epoch%s.h5" % epoch)
    }
    return paths


# 加载模型的权重
def load_model_state(gen, disc, gan, opt_disc, opt_gan, model_name, epoch):
    # 获取权重文件路径
    paths = model_state_paths(model_name, epoch)
    # 加载生成器和判别器的权重
    gen.load_weights(paths["gen_weights_path"])
    disc.load_weights(paths["disc_weights_path"])

    # 设置判别器不可训练，避免在训练GAN时更新判别器
    disc.trainable = False
    gan.compile(optimizer=opt_gan, loss='binary_crossentropy') # 编译GAN网络
    # gan._make_train_function()
    data_utils.load_opt_weights(gan, paths["opt_gan_weights_path"])  # 加载GAN的优化器权重
    # 恢复判别器训练状态，并编译判别器
    disc.trainable = True
    disc.compile(optimizer=opt_disc, loss='binary_crossentropy')
    data_utils.load_opt_weights(disc, paths["opt_disc_weights_path"])  # 加载判别器的优化器权重




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







