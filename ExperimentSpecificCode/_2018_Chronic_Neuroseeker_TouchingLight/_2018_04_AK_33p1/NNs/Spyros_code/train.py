
from os.path import join
import numpy as np
import pandas as pd
import time

import cv2
import matplotlib.pyplot as plt
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


def build_subsampling():
    from keras.layers import Input, AveragePooling2D, MaxPooling2D
    from keras.models import Model

    # This returns a tensor
    input = Input(shape=(360, 838,1))
    #x = AveragePooling2D(pool_size=(4,5))(input)
    x = MaxPooling2D(pool_size=(4, 5))(input)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam',
                  loss='mse')

    #print(model.summary())

    return model

def build_network_split(spike_shape, image_shape):
    from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, MaxPooling2D
    from keras.layers import CuDNNLSTM
    from keras.models import Model
    from keras.optimizers import Adam

    # This returns a tensor
    input_0 = Input(shape=(spike_shape[1],spike_shape[2],spike_shape[3]))
    reshaped_input = Reshape(target_shape = (spike_shape[1],spike_shape[2]))(input_0)
    input_1 = Input(shape=(1,image_shape[2], image_shape[3]))


    ## let's start with 2D convolutions 3 X 3

    #x_spikes = Convolution2D(filters=10, kernel_size=(3,3), activation="elu")(input_0)
    #x_spikes = BatchNormalization()(x_spikes)
    #x_spikes = MaxPooling2D(pool_size=(4, 5))(x_spikes)
    x_spikes = CuDNNLSTM(32)(reshaped_input)
    #x_spikes = Dropout(0.1)(x_spikes)
    # x_spikes = MaxPooling2D(pool_size=(4, 10))(x_spikes)
    # x_spikes = Convolution2D(filters=3, kernel_size=(3,3), activation="elu")(x_spikes)
    # x_spikes = BatchNormalization()(x_spikes)
    #x_spikes = Flatten()(x_spikes)
    #x_spikes = Dense(64, activation='relu')(x_spikes)
    #x_spikes = BatchNormalization()(x_spikes)

    x_image = Convolution2D(filters=4, kernel_size=(3, 3), activation="elu", data_format='channels_first')(input_1)
    #x_image = BatchNormalization(axis = 1)(x_image)
    #x_image = Dropout(0.1)(x_image)
    x_image = Flatten()(x_image)
    #x_image = Dense(64, activation='relu')(x_image)
    #x_image = BatchNormalization()(x_image)


    x = concatenate([x_image, x_spikes])

    #x = x_image
    #x = concatenate([x, x_spikes])

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)


    predictions = Dense(image_shape[2]*image_shape[3], activation='sigmoid')(x)
    reshaped_pred = Reshape((image_shape[2],image_shape[3]), name = "image_loss")(predictions)

    predictions_spike = concatenate([predictions, x_spikes])
    predictions_spike = Dropout(0.5)(predictions_spike)

    predictions_spike  = Dense(image_shape[2]*image_shape[3], activation='sigmoid')(predictions_spike)
    predictions_spike = Reshape((image_shape[2],image_shape[3]), name = "spike_loss")(predictions_spike)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[input_0, input_1], outputs=[reshaped_pred, predictions_spike])
    model.compile(optimizer=Adam(lr=0.001),
                  loss='mse')

    #print(model.summary())

    return model



def build_network(spike_shape, image_shape, with_spikes=True ):
    from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, MaxPooling2D, CuDNNLSTM, LSTM, dot
    from keras.models import Model
    from keras.optimizers import Adam

    # This returns a tensor
    input_0 = Input(shape=(spike_shape[1],spike_shape[2],spike_shape[3]))
    input_1 = Input(shape=(1,image_shape[1], image_shape[2]))


    ## let's start with 2D convolutions 3 X 3
    reshaped_input = Reshape(target_shape=(spike_shape[1], spike_shape[2]))(input_0)
    x_spikes = CuDNNLSTM(64)(reshaped_input)
    x_image = Convolution2D(filters=4, kernel_size=(3, 3), activation="elu", data_format='channels_first')(input_1)
    x_image = Flatten()(x_image)


    x = x_image

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)
    if with_spikes:
        x = concatenate([x, x_spikes])
    predictions = Dense(image_shape[1]*image_shape[2], activation='sigmoid')(x)

    predictions = Reshape((image_shape[1], image_shape[2]))(predictions)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[input_0, input_1], outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0005),
                  loss='mse')

    #print(model.summary())

    return model


#
base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
data_folder = join(base_data_folder, 'Data')
with np.load(join(data_folder, "data_random_5secs_data_10secs_images_half_size_res.npz")) as data:
    X = data['X']
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = data["Y"]/255.0
    starting_images = Y[:,0:1,:,:]
    ending_images = Y[:, 1, :, :]
    r = data["r"]
sort_idx = r.argsort()
#print(sort_idx)
print(X.shape)
print(starting_images.shape)
print(ending_images.shape)
#print(r[sort_idx])
#X = X[sort_idx]
#starting_images = starting_images[sort_idx]
#ending_images = ending_images[sort_idx]

# SUBSAMPLE

#sampler = build_subsampling()
#X = sampler.predict(X)
print(X.shape)
print(X.max())

X_train, X_test, starting_images_train, starting_images_test, ending_images_train, ending_images_test \
    = train_test_split(X,starting_images, ending_images, shuffle=False, test_size=0.10 )



print(X_train.shape)

model_spiky = build_network(X.shape, ending_images.shape, True)
print(model_spiky.summary())


spiky_checkpoint_file = (join(data_folder, 'spiky_latest_model_half_size_spyros.h5'))
spiky_checkpoint = ModelCheckpoint(spiky_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
spiky_callbacks_list = [spiky_checkpoint]
model_spiky.fit([X_train, starting_images_train], ending_images_train,
          validation_data=([X_test, starting_images_test], ending_images_test),
          epochs=300, callbacks=spiky_callbacks_list)
model_spiky.save(join(data_folder, 'spiky_model_half_size_spyros.h5'))

#model.fit([X_train,starting_images_train],[ending_images_train,ending_images_train],
#          validation_data=([X_test, starting_images_test], [ending_images_test, ending_images_test]),epochs = 10000  )


# print(X.shape)
# print(Y.shape)

model_pictures = build_network(X.shape, ending_images.shape, False)
print(model_pictures.summary())

pictures_checkpoint_file = (join(data_folder, 'pictures_latest_model_half_size_spyros.h5'))
pictures_checkpoint = ModelCheckpoint(pictures_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
pictures_callbacks_list = [pictures_checkpoint]
model_pictures.fit([X_train, starting_images_train], ending_images_train,
          validation_data=([X_test, starting_images_test], ending_images_test),
          epochs=300, callbacks = pictures_callbacks_list)
model_pictures.save(join(data_folder, 'pictures_model_half_size_spyros.h5'))

