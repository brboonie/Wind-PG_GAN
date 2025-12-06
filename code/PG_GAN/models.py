from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Flatten, Input
from keras.layers import LeakyReLU, Reshape, ReLU
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from structure import *
import numpy as np


# completion network
def cs_generator(scene_size, modis_var_dim):
    # 输入层
    modis_scene_input = Input(shape=(scene_size, scene_size, 1), name="modis_scene_in")
    modis_var_input = Input(shape=(scene_size, scene_size, modis_var_dim), name="modis_var_in")
    mask_input = Input(shape=(scene_size, scene_size, 1), name="modis_mask_in")


    # 第一层卷积
    conv1 = Conv2D(64, kernel_size=5, strides=1, padding="SAME", name="conv1")(modis_var_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    # 第二层卷积
    conv2 = Conv2D(128, kernel_size=3, strides=2, padding="SAME", name="conv2")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    # 第三层卷积
    conv3 = Conv2D(128, kernel_size=3, strides=1, padding="SAME", name="conv3")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    # 第四层卷积
    conv4 = Conv2D(256, kernel_size=3, strides=2, padding="SAME", name="conv4")(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)

    # 第五层卷积
    conv5 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", name="conv5")(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)

    # 第六层卷积
    conv6 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", name="conv6")(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    # Dilated convolution 层
    dilate_conv1 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", dilation_rate=2, name="dilate_conv1")(
        conv6)
    dilate_conv1 = ReLU()(dilate_conv1)

    dilate_conv2 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", dilation_rate=4, name="dilate_conv2")(
        dilate_conv1)
    dilate_conv2 = ReLU()(dilate_conv2)

    dilate_conv3 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", dilation_rate=8, name="dilate_conv3")(
        dilate_conv2)
    dilate_conv3 = ReLU()(dilate_conv3)

    dilate_conv4 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", dilation_rate=16, name="dilate_conv4")(
        dilate_conv3)
    dilate_conv4 = ReLU()(dilate_conv4)

    # 恢复卷积层
    conv7 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", name="conv7")(dilate_conv4)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)

    conv8 = Conv2D(256, kernel_size=3, strides=1, padding="SAME", name="conv8")(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    # 反卷积层1
    deconv1 = Conv2DTranspose(128, kernel_size=3, strides=2, padding="SAME", output_padding=1, name="deconv1")(
        conv8)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = ReLU()(deconv1)

    # 卷积层9
    conv9 = Conv2D(128, kernel_size=3, strides=1, padding="SAME", name="conv9")(deconv1)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)

    # 反卷积层2
    deconv2 = Conv2DTranspose(64, kernel_size=3, strides=2, padding="SAME", output_padding=1, name="deconv2")(
        conv9)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = ReLU()(deconv2)

    # 卷积层10
    conv10 = Conv2D(32, kernel_size=3, strides=1, padding="SAME", name="conv10")(deconv2)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)

    # 最后一层卷积
    conv11 = Conv2D(2, kernel_size=3, strides=1, padding="SAME", name="conv11")(conv10)
    conv11 = BatchNormalization()(conv11)
    output = Activation("tanh")(conv11)

    # 创建模型
    gen = Model(inputs=modis_var_input, outputs=output, name="gen")
    return gen



def linear(input, output_size, name="linear"):
    with tf.compat.v1.variable_scope(name):
        shape = input.get_shape().as_list()
        matrix = tf.compat.v1.get_variable("W", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.compat.v1.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, matrix) + bias

def discriminator(scene_size, modis_var_dim):

    nets = []

    disc_input = Input(shape=(scene_size,scene_size,modis_var_dim+4), name="disc_in")

    conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv1")(disc_input)
    conv1 = tf.keras.layers.BatchNormalization(name="bn1")(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    nets.append(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv2")(conv1)
    conv2 = tf.keras.layers.BatchNormalization(name="bn2")(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    nets.append(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv3")(conv2)
    conv3 = tf.keras.layers.BatchNormalization(name="bn3")(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    nets.append(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv4")(conv3)
    conv4 = tf.keras.layers.BatchNormalization(name="bn4")(conv4)
    conv4 = tf.keras.layers.ReLU()(conv4)
    nets.append(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv5")(conv4)
    conv5 = tf.keras.layers.BatchNormalization(name="bn5")(conv5)
    conv5 = tf.keras.layers.ReLU()(conv5)
    nets.append(conv5)

    conv6 = tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same", activation=None, name="conv6")(conv5)
    conv6 = tf.keras.layers.BatchNormalization(name="bn6")(conv6)
    conv6 = tf.keras.layers.ReLU()(conv6)
    nets.append(conv6)

    flatten = tf.keras.layers.Flatten()(conv6)
    output = linear(flatten, 1024, name="linear")

    model = Model(inputs=[disc_input], outputs=output, name="disc")

    return model


def calc_loss(logits, label):
    if label == 1:
        y = tf.ones_like(logits)
    else:
        y = tf.zeros_like(logits)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))


def cs_modis_cgan(gen, disc, scene_size, modis_var_dim):
    modis_var_input = Input(shape=(scene_size,scene_size,modis_var_dim),
        name="modis_var_in")
    modis_mask_input = Input(shape=(scene_size,scene_size,1), name="modis_mask_in")
    modis_scene = Input(shape=(scene_size,scene_size,1),name="modis_scene_in")

    generated_image = gen(modis_var_input)
    generated_image_first_channel = generated_image[..., 0:1]
    # disc_inputs = Concatenate(axis=-1)([generated_image, generated_image_first_channel])
    disc_inputs = Concatenate(axis=-1)([generated_image, generated_image, generated_image])
    # disc_inputs = generated_image
    x_disc = disc(disc_inputs)

    gan = Model(inputs=modis_var_input, outputs=x_disc, name="cs_modis_cgan")

    return gan


# Load the pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False)

# Define the layers to extract features from
layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
layers = [vgg16.get_layer(name).output for name in layer_names]

# Create a model that outputs the selected layers
feature_extractor = Model(inputs=vgg16.input, outputs=layers)



# Define the function to extract features
def extract_features(image):
    if image.shape[-1] == 1:
        image_resized = tf.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 2:
        # 计算第三个通道为第一个通道的平方 + 第二个通道的平方，再开根号
        third_channel = tf.sqrt(tf.square(image[..., 0]) + tf.square(image[..., 1]))
        # 拼接三个通道
        image_resized = tf.concat([image, third_channel[..., tf.newaxis]], axis=-1)

    features = feature_extractor(image_resized)
    return features



# perceptual loss
def calculate_perceptual_loss(observations, reconstructions):
    obs_features = extract_features(observations)
    recon_features = extract_features(reconstructions)

    perceptual_loss = 0
    for obs_feat, recon_feat in zip(obs_features, recon_features):
        perceptual_loss += tf.reduce_mean(tf.square(obs_feat - recon_feat))

    return perceptual_loss

def discriminator_loss(logits_real_u, logits_fake_u):
    # Calculate the discriminator loss
    real_loss = tf.reduce_mean(tf.math.log(tf.sigmoid(logits_real_u) + 1e-8))
    fake_loss = tf.reduce_mean(tf.math.log(1.0 - tf.sigmoid(logits_fake_u) + 1e-8))
    loss = -(real_loss + fake_loss)
    return loss

def generator_loss(logits_fake_u):
    # Calculate the generator loss
    gen_loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(logits_fake_u) + 1e-8))
    return gen_loss


def calculate_parallel(u, v, u_true, v_true):
    # 计算 u × v_true
    uv_cross = u * v_true - v * u_true  # 二维叉积公式

    # 返回叉积差（parallel）
    return tf.reduce_sum(tf.square(uv_cross))  # 求和以得到标量值


def calculate_speed(u, v, u_true, v_true):
    # 计算 u × v_true
    speed_gen = tf.sqrt(u**2 + v**2)
    speed_true = tf.sqrt(u_true**2 + v_true**2)

    # 返回叉积差（parallel）
    return tf.reduce_sum(tf.square(speed_true - speed_gen))  # 求和以得到标量值

def calculate_gradient(u, v, u_true, v_true):

    dx= 0.03125
    dy= 0.03125

    dU_dx = np.zeros_like(u)
    dU_dy = np.zeros_like(u)
    dUtrue_dx = np.zeros_like(u_true)
    dUtrue_dy = np.zeros_like(u_true)

    for t in range(u.shape[0]):
        dU_dx[t, :, :] = np.gradient(u[t, :, :], dx, axis=0)
        dU_dy[t, :, :] = np.gradient(u[t, :, :], dy, axis=1)
        dUtrue_dx[t, :, :] = np.gradient(u_true[t, :, :], dx, axis=0)
        dUtrue_dy[t, :, :] = np.gradient(u_true[t, :, :], dy, axis=1)

    dV_dx = np.zeros_like(v)
    dV_dy = np.zeros_like(v)
    dVtrue_dx = np.zeros_like(v_true)
    dVtrue_dy = np.zeros_like(v_true)

    for t in range(u.shape[0]):
        dV_dx[t, :, :] = np.gradient(v[t, :, :], dx, axis=0)
        dV_dy[t, :, :] = np.gradient(v[t, :, :], dy, axis=1)
        dVtrue_dx[t, :, :] = np.gradient(v_true[t, :, :], dx, axis=0)
        dVtrue_dy[t, :, :] = np.gradient(v_true[t, :, :], dy, axis=1)

    grad_ux = dU_dx - dUtrue_dx
    grad_uy = dU_dy - dUtrue_dy
    grad_vx = dV_dx - dVtrue_dx
    grad_vy = dV_dy - dVtrue_dy

    return tf.reduce_sum((tf.square(grad_ux)+tf.square(grad_uy)+tf.square(grad_vx)+tf.square(grad_vy))/4)







