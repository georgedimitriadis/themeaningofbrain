
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, \
    MaxPooling2D, AveragePooling2D, CuDNNLSTM, LSTM, dot
from keras.models import Model, load_model
from keras.optimizers import Adam


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


def generator(X, starting_images, ending_images, train_index, batch_size=500):
    i = 0
    num_of_samples = train_index.shape[0]
    one_extra = 0
    if num_of_samples % batch_size:
        one_extra = 1
    steps_per_epoch = num_of_samples // batch_size + one_extra

    while True:
        indices = train_index[batch_size*i : batch_size*(i+1)]
        if batch_size * (i+1) > train_index.shape[0]:
            indices = train_index[batch_size*i : train_index.shape[0]]
        if steps_per_epoch == i + 1:
            i = 0
        else:
            i = i+1
        yield [X[indices], starting_images[indices]], ending_images[indices]


def generator_random(X, starting_images, ending_images, train_index, batch_size=500):
    while True:
        indices = np.random.choice(train_index, batch_size, replace=False)
        yield [X[indices], starting_images[indices]], ending_images[indices]


base_data_folders = {'NS': r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs',
                     'Long': r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs',
                     'Sparse': r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs'}

# -------- USER INPUT ----------
run_with = ['Image']  # ['Both', 'Spikes', 'Image']
base_data_folder = base_data_folders['NS']  # 'NS', or 'Long' or 'Sparse'
data_folder_name = 'data_100KsamplesEvery2Frames_5secslong_halfsizeres'
n_splits = 10
starting_iter = 1
# ------------------------------

data_folder = join(base_data_folder, 'Data', 'TimeSeriesSplit', data_folder_name)
input_data_name_X = "X_buffer.npy"
input_data_name_Y = "Y_buffer.npy"

headers = np.load(join(data_folder, 'binary_headers.npz'), allow_pickle=True)
X = np.memmap(join(data_folder, input_data_name_X), dtype=headers['dtype'][0], shape=tuple(headers['shape_X']))
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

Y = np.memmap(join(data_folder, input_data_name_Y), dtype=headers['dtype'][0], shape=tuple(headers['shape_Y']))
starting_images = Y[:, 0:1, :, :]/255.0
ending_images = Y[:, 1, :, :]/255.0

print(X.shape)
print(starting_images.shape)
print(ending_images.shape)

frames_used = headers['shape_X'][1]

tscv = TimeSeriesSplit(gap=frames_used, max_train_size=None, n_splits=n_splits, test_size=None)

i = 0
histories_of_losses = []
histories_of_val_losses = []


for train_index, test_index in tscv.split(X):
    print(len(train_index))

    if i < starting_iter:
        i = i+1
        continue

    print('TRAIN from {} to {}, TEST from {} to {}'.format(train_index[0], train_index[-1], test_index[0], test_index[-1]))

    X_test = X[test_index]
    starting_images_train, starting_images_test = starting_images[train_index], starting_images[test_index]
    ending_images_train, ending_images_test = ending_images[train_index], ending_images[test_index]

    batch_size = 500
    gen = generator_random(X, starting_images_train, ending_images_train, train_index, batch_size=batch_size)
    num_of_samples = train_index.shape[0]
    one_extra = 0
    if num_of_samples % batch_size:
        one_extra = 1
    steps_per_epoch = num_of_samples // batch_size + one_extra
    print('STEPS PER EPOCH = '.format(steps_per_epoch))

    if 'Both' in run_with:
        model_full = build_network(X.shape, ending_images.shape, spikes_images_type='Both')
        print(model_full.summary())

        full_checkpoint_file = (join(data_folder, 'both_latest_model_SSTiter_{}.h5'.format(i)))
        full_checkpoint = ModelCheckpoint(full_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        full_callbacks_list = [full_checkpoint]
        model_history = model_full.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                                                 validation_data=([X_test, starting_images_test], ending_images_test),
                                                 epochs=300, callbacks=full_callbacks_list)
        model_full.save(join(data_folder, 'both_final_model_SSTiter_{}.h5'.format(i)))

    if 'Spikes' in run_with:
        model_spikes = build_network(X.shape, ending_images.shape, spikes_images_type='Spikes')
        print(model_spikes.summary())

        spikes_checkpoint_file = (join(data_folder, 'spikes_latest_model_SSTiter_{}.h5'.format(i)))
        spikes_checkpoint = ModelCheckpoint(spikes_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        spikes_callbacks_list = [spikes_checkpoint]
        model_history = model_spikes.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                                                   validation_data=([X_test, starting_images_test], ending_images_test),
                                                   epochs=300, callbacks=spikes_callbacks_list)
        model_spikes.save(join(data_folder, 'spikes_final_model_SSTiter_{}.h5'.format(i)))

    if 'Image' in run_with:
        model_pictures = build_network(X.shape, ending_images.shape, spikes_images_type='Image')
        print(model_pictures.summary())

        pictures_checkpoint_file = (join(data_folder, 'pictures_latest_model_SSTiter_{}.h5'.format(i)))
        pictures_checkpoint = ModelCheckpoint(pictures_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        pictures_callbacks_list = [pictures_checkpoint]

        model_history = model_pictures.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                                                     validation_data=([X_test, starting_images_test], ending_images_test),
                                                     epochs=300, callbacks=pictures_callbacks_list)

        model_pictures.save(join(data_folder, 'pictures_final_model_SSTiter_{}.h5'.format(i)))

    histories_of_losses.append(model_history.history['loss'])
    histories_of_val_losses.append(model_history.history['val_loss'])

    i = i+1

    np.save(join(data_folder, 'loss_histories_of_{}.npy'.format(run_with[0])), np.array(histories_of_losses))
    np.save(join(data_folder, 'val_loss_histories_of_{}.npy'.format(run_with[0])), np.array(histories_of_val_losses))


#  Visualise results
'''
loss = np.load(join(data_folder, 'loss_histories_of_Spikes.npy'))
val_loss = np.load(join(data_folder, 'val_loss_histories_of_Spikes.np'))

import h5py

model_5_filename = join(data_folder, "spikes_final_model_SSTiter_5.h5")

model_5_h5 = h5py.File(model_5_filename,'w')

data_p = model_5_h5.attrs['training_config']
data_p = data_p.decode().replace("learning_rate","lr").encode()
model_5_h5.attrs['training_config'] = data_p
model_5_h5.close()
'''