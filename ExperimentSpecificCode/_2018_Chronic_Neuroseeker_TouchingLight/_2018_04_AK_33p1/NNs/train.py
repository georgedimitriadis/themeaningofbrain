
import numpy as np
import pickle
from constants import *

# from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1.NNs.constants import *


from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, \
    MaxPooling2D, AveragePooling2D, CuDNNLSTM, LSTM, dot
from keras.optimizers import Adam

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split


def build_network(spike_shape, image_shape, with_spikes=True):

    # This returns a tensor
    input_0 = Input(shape=(spike_shape[1], spike_shape[2]))
    input_1 = Input(shape=(1, image_shape[1], image_shape[2]))

    ## let's start with 2D convolutions 3 X 3
    x_spikes = CuDNNLSTM(64)(input_0)
    x_image = Convolution2D(filters=4, kernel_size=(3, 3), activation="elu", data_format='channels_first')(input_1)
    x_image = Flatten()(x_image)

    x = x_image

    if with_spikes:
        x = concatenate([x, x_spikes])

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(image_shape[1]*image_shape[2], activation='sigmoid')(x)

    predictions = Reshape((image_shape[1], image_shape[2]))(predictions)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[input_0, input_1], outputs=predictions)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='mse')

    #print(model.summary())

    return model


"""
with np.load("data_random-full.npz") as data:
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
    = train_test_split(X, starting_images, ending_images, shuffle=False, test_size=0.10)
"""

data = np.load(save_data_file)
X_train = data['X_train']
X_test = data['X_test']
starting_images_train = data['starting_images_train']
starting_images_test = data['starting_images_test']
ending_images_train = data['ending_images_train']
ending_images_test = data['ending_images_test']


spiky_model = build_network(X_train.shape, ending_images_train.shape, with_spikes=True)
print(spiky_model.summary())

spiky_model.fit([X_train,starting_images_train],ending_images_train,
          validation_data=([X_test, starting_images_test], ending_images_test),
          epochs=300)

spiky_model.save(spiky_model_file)

picture_model = build_network(X_train.shape, ending_images_train.shape, with_spikes=False)
print(picture_model.summary())

picture_model.fit([X_train, starting_images_train], ending_images_train,
          validation_data=([X_test, starting_images_test], ending_images_test),
          epochs=300)

picture_model.save(picture_model_file)




