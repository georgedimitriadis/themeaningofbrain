
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, \
    MaxPooling2D, AveragePooling2D, CuDNNLSTM, LSTM, dot
from keras.models import Model
from keras.optimizers import Adam

data_folder_suffix = 'with_poisson_25Krandom_2secs'

input_data_name = "data_extra_poisson_25000randompoints_2secslong_halfsizeres.npz"

run_with = ['Spikes']  # ['Both', 'Spikes', 'Image']


def build_network(spike_shape, image_shape, spikes_images_type='Both'): # type = Both OR Spikes OR Image

    input_0 = Input(shape=(spike_shape[1], spike_shape[2], spike_shape[3]))
    input_1 = Input(shape=(1, image_shape[1], image_shape[2]))

    reshaped_input = Reshape(target_shape=(spike_shape[1], spike_shape[2]))(input_0)
    x_spikes = CuDNNLSTM(32)(reshaped_input)

    x_image = Convolution2D(filters=4, kernel_size=(3, 3), activation="elu", data_format='channels_first')(input_1)
    x_image = Flatten()(x_image)
    x_image = Dense(1024, activation='elu')(x_image)
    x_image = Dropout(0.5)(x_image)
    x_image = Dense(1024, activation='elu')(x_image)
    x_image = Dropout(0.5)(x_image)

    if spikes_images_type == 'Both':
        x = concatenate([x_image, x_spikes])
    elif spikes_images_type == 'Spikes':
        x = x_spikes
    elif spikes_images_type == 'Image':
        x = x_image

    predictions = Dense(image_shape[1]*image_shape[2], activation='sigmoid')(x)
    predictions = Reshape((image_shape[1], image_shape[2]))(predictions)

    model = Model(inputs=[input_0, input_1], outputs=predictions)
    model.compile(optimizer=Adam(lr=0.0005),
                  loss='mse')
    return model


base_data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NNs'
#base_data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NeuropixelSimulations\Long\NNs'
#base_data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NeuropixelSimulations\Sparce\NNs'
data_folder = join(base_data_folder, 'Data', 'RandomisedInput')

with np.load(join(data_folder, input_data_name)) as data:
    X = data['X']
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = data["Y"]/255.0
    starting_images = Y[:, 0:1, :, :]
    ending_images = Y[:, 1, :, :]
    r = data["r"]
sort_idx = r.argsort()

print(X.shape)
print(X.max())
print(starting_images.shape)
print(ending_images.shape)


X_train, X_test, starting_images_train, starting_images_test, ending_images_train, ending_images_test \
    = train_test_split(X, starting_images, ending_images, shuffle=False, test_size=0.10 )
print(X_train.shape)


if 'Both' in run_with:
    model_full = build_network(X.shape, ending_images.shape, spikes_images_type='Both')
    print(model_full.summary())

    full_checkpoint_file = (join(data_folder, 'both_latest_model_{}.h5'.format(data_folder_suffix)))
    full_checkpoint = ModelCheckpoint(full_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True,
                                       mode='min')
    full_callbacks_list = [full_checkpoint]
    model_full.fit([X_train, starting_images_train], ending_images_train,
                    validation_data=([X_test, starting_images_test], ending_images_test),
                    epochs=300, callbacks=full_callbacks_list)
    model_full.save(join(data_folder, 'both_model_{}.h5'.format(data_folder_suffix)))

if 'Spikes' in run_with:
    model_spikes = build_network(X.shape, ending_images.shape, spikes_images_type='Spikes')
    print(model_spikes.summary())

    spikes_checkpoint_file = (join(data_folder, 'spikes_latest_model_{}.h5'.format(data_folder_suffix)))
    spikes_checkpoint = ModelCheckpoint(spikes_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    spikes_callbacks_list = [spikes_checkpoint]
    model_spikes.fit([X_train, starting_images_train], ending_images_train,
              validation_data=([X_test, starting_images_test], ending_images_test),
              epochs=300, callbacks=spikes_callbacks_list)
    model_spikes.save(join(data_folder, 'spikes_model_{}.h5'.format(data_folder_suffix)))


if 'Image' in run_with:
    model_pictures = build_network(X.shape, ending_images.shape, spikes_images_type='Image')
    print(model_pictures.summary())

    pictures_checkpoint_file = (join(data_folder, 'pictures_latest_model_{}.h5'.format(data_folder_suffix)))
    pictures_checkpoint = ModelCheckpoint(pictures_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    pictures_callbacks_list = [pictures_checkpoint]
    model_pictures.fit([X_train, starting_images_train], ending_images_train,
              validation_data=([X_test, starting_images_test], ending_images_test),
              epochs=300, callbacks = pictures_callbacks_list)
    model_pictures.save(join(data_folder, 'pictures_model_{}.h5'.format(data_folder_suffix)))

