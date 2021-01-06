# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:50:34 2018

@author: Dima's_Monster
"""

import keras
from keras.layers import Conv2D, UpSampling2D, concatenate, Input, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2, l1

import numpy as np

import os

import cv2

import sys

import matplotlib.pyplot as plt

weights_decay = 5e-4



## VARIABLES TO CHANGE ##
loading_weights = True
loading_combined = True
weights = 'WEIGHTSFILENAME.h5'
number_epochs = 150
batch_size = 4


# optimizer
#adam = Adam(lr=1e-3, beta_1=0.5, beta_2=0.999, decay=0.9, amsgrad=True, clipnorm=1.0)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# loss weights
loss_weights = [1e2, 1]

bn_axis = -1

im_size = 256

### DATA ###
# getData
x_path = './test_data/input'
y_path = './test_data/target'
p_path = './test_data/test_images'

# training / plotting outcomes / saving images locally
training = False
plotting = False
saving = True

def get_paths(folder_path, files):
    for i in range(len(files)):
        files[i] = folder_path + files[i]
    return files

x_images = os.listdir(x_path)
y_images = os.listdir(y_path)
p_images = os.listdir(p_path)

X_train = []
y_train = []

p_imgs = []

for i in range(len(x_images)):
    pth = os.path.join(x_path, x_images[i])
    imgx = cv2.imread(pth)
    imgx = cv2.resize(imgx, (im_size,im_size))
    imgx = cv2.bitwise_not(imgx)
    X_train.append(imgx)
    pth = os.path.join(y_path, y_images[i])
    imgy = cv2.imread(pth)
    imgy = cv2.resize(imgy, (im_size,im_size))
    imgy = imgy[:,:,[2,1,0]]
    y_train.append(imgy)

for i in range(len(p_images)):
    pth=os.path.join(p_path, p_images[i])
    imgx=cv2.imread(pth)
    imgx = cv2.resize(imgx, (im_size,im_size))
    imgx = cv2.bitwise_not(imgx)
    p_imgs.append(imgx)

X_train = (np.asarray(X_train) - 127.5) / 127.5
y_train = (np.asarray(y_train) - 127.5) / 127.5
p_imgs = (np.asarray(p_imgs) - 127.5) / 127.5


k_size = 4

p_e_save = 1

d_shape = (im_size,im_size,3)

data_size = X_train.shape[0] # how many units of data
batch_count = int(np.ceil(data_size / batch_size))

#half_batch = int(batch_size / 2)
half_batch = 2

# Unet model
input_img = Input(shape = d_shape)
# 256 x 256
x1 = Conv2D(64, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(input_img)
x1 = LeakyReLU(0.2)(x1)
x1 = BatchNormalization( axis = bn_axis)(x1)
# 128 x 128
x2 = Conv2D(128, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x1)
x2 = LeakyReLU(0.2)(x2)
x2 = BatchNormalization( axis = bn_axis)(x2)
# 64 x 64
x3 = Conv2D(256, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x2)
x3 = LeakyReLU(0.2)(x3)
x3 = BatchNormalization( axis = bn_axis)(x3)
# 32 x 32
x4 = Conv2D(512, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x3)
x4 = LeakyReLU(0.2)(x4)
x4 = BatchNormalization( axis = bn_axis)(x4)
# 16 x 16
x5 = Conv2D(512, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x4)
x5 = LeakyReLU(0.2)(x5)
x5 = BatchNormalization( axis = bn_axis)(x5)
# 8 x 8
x6 = Conv2D(512, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x5)
x6 = LeakyReLU(0.2)(x6)
x6 = BatchNormalization( axis = bn_axis)(x6)
# 4 x 4
x7 = Conv2D(512, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x6)
x7 = LeakyReLU(0.2)(x7)
x7 = BatchNormalization( axis = bn_axis)(x7)
# 2 x 2
x8 = Conv2D(512, (k_size, k_size), strides=(2,2), padding='same', kernel_regularizer=l2(weights_decay))(x7)
x8 = LeakyReLU(0.2)(x8)
x8 = BatchNormalization( axis = bn_axis)(x8)
# 1 x 1
y1 = UpSampling2D()(x8)
y1 = Conv2D(512, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y1)
y1 = BatchNormalization( axis = bn_axis)(y1)
# 2 x 2

y2 = concatenate([x7, y1], axis = -1)
y2 = UpSampling2D()(y2)
y2 = Conv2D(512, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y2)
y2 = BatchNormalization( axis = bn_axis)(y2)

# 4 x 4
y3 = concatenate([x6, y2], axis = -1)
y3 = UpSampling2D()(y3)
y3 = Conv2D(512, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y3)
y3 = BatchNormalization( axis = bn_axis)(y3)

# 8 x 8
y4 = concatenate([x5, y3], axis = -1)
y4 = UpSampling2D()(y4)
y4 = Conv2D(512, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y4)
y4 = BatchNormalization( axis = bn_axis)(y4)

# 16 x 16
y5 = concatenate([x4, y4], axis = -1)
y5 = UpSampling2D()(y5)
y5 = Conv2D(512, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y5)
y5 = BatchNormalization( axis = bn_axis)(y5)

# 32 x 32
y6 = concatenate([x3, y5], axis = -1)
y6 = UpSampling2D()(y6)
y6 = Conv2D(256, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y6)
y6 = BatchNormalization( axis = bn_axis)(y6)

# 64 x 64
y7 = concatenate([x2, y6], axis = -1)
y7 = UpSampling2D()(y7)
y7 = Conv2D(128, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y7)
y7 = BatchNormalization( axis = bn_axis)(y7)

# 128 x 128
y8 = concatenate([x1, y7], axis = -1)
y8 = UpSampling2D()(y7)
y8 = Conv2D(64, (k_size, k_size), activation='relu', padding='same', kernel_regularizer=l2(weights_decay))(y8)
y8 = BatchNormalization( axis = bn_axis)(y8)
# 256 x 256

# 256 x 256
output_img = Conv2D(3, (k_size, k_size), activation='tanh', padding='same', kernel_regularizer=l2(weights_decay))(y8)

pix2pix = Model(input_img, output_img)
pix2pix.compile(loss='mae', optimizer=adam)


# Discriminator model
disc_input = Input(shape=d_shape)
# 256 x 256
d = Conv2D(64, (k_size, k_size), activation='relu', padding='same', strides=(2,2))(disc_input)
d = BatchNormalization()(d)
# 128 x 128
d = Conv2D(128, (k_size, k_size), activation='relu', padding='same', strides=(2,2))(d)
d = BatchNormalization()(d)
# 64 x 64
d = Conv2D(256, (k_size, k_size), activation='relu', padding='same', strides=(2,2))(d)
d = BatchNormalization()(d)
# 32 x 32
d = Conv2D(512, (k_size, k_size), activation='relu', padding='same', strides=(2,2))(d)
d = BatchNormalization()(d)
# 16 x 16

disc_output = Conv2D(1, (k_size, k_size), activation='sigmoid', padding='same')(d)

discriminator = Model(disc_input, disc_output)
discriminator.compile(optimizer=adam, loss='binary_crossentropy')

discriminator.trainable = False

# GAN (combined)
gan_input = Input(shape=d_shape)
generated_imgs = pix2pix(gan_input)
gan_output = discriminator(generated_imgs)

combined = Model(gan_input, [generated_imgs, gan_output])
combined.compile(optimizer=adam, loss=['mae', 'binary_crossentropy'], loss_weights=loss_weights)


if loading_weights:
    try:
        if loading_combined:
            combined.load_weights(weights)
        else:
            pix2pix.load_weights(weights)
    except:
        print('No weights file detected. Proceeding without loading weights')
            
epoch_d_loss = []
epoch_g_loss = []
step_count = []

if training == True:
    for epoch in range(number_epochs):

        for step in range(batch_count):
            index = np.random.randint(0, X_train.shape[0], size=half_batch)
            src_images = X_train[index]
            
            generated_images = pix2pix.predict(src_images)
            
            # patch labels
            real_y = np.ones((half_batch, 16,16,1)) * 0.9
            fake_y = real_y * 0.1
            
            # train discriminator
            d_loss_r = discriminator.train_on_batch(src_images, real_y)
            d_loss_f = discriminator.train_on_batch(generated_images, fake_y)
            
            d_loss = (d_loss_r + d_loss_f) / 2
            epoch_d_loss.append(d_loss)
            
            # train GAN
            index = np.random.randint(0, X_train.shape[0], size=batch_size)
            src_images = X_train[index]
            trgt_images = y_train[index]
            
            real_y = np.ones((batch_size, 16, 16,1))
            
            g_loss_l = combined.train_on_batch(src_images ,[trgt_images, real_y])
            
            g_loss = np.average(g_loss_l)
            
            epoch_g_loss.append(g_loss)
          
            step_count.append(step+1)
            print('d_loss {}, g_loss {}\nepoch {} is {:2f}% complete'.format(d_loss, g_loss, epoch+1, step/batch_count * 100))
        if (epoch + 1) % p_e_save == 0:
            pix2pix.save_weights(weights)
            combined.save_weights(weights)

# encoder decoder
predicted_images = pix2pix.predict(p_imgs)
#predicted_images = pix2pix.predict(X_train)


if plotting == True:
    # show 10 images from decoder
    n = 12  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        tst = ((y_train[i] * 127.5) + 127.5) / 255
        plt.imshow(tst)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        prd = ((predicted_images[i] * 127.5) + 127.5) / 255
        plt.imshow(prd)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if saving:
    for i in range(predicted_images.shape[0]):
        img = predicted_images[i,:,:,:]
        img = (img * 127.5) + 127.5
        cv2.imwrite('iter_{}.png'.format(i), img)
    