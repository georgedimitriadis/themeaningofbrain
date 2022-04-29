
from os.path import join
import numpy as np
import argparse
import timeit

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, BatchNormalization, Dropout, \
    MaxPooling2D, AveragePooling2D, CuDNNLSTM, LSTM, dot
from keras.models import Model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split, TimeSeriesSplit


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

def get_args():
    """ Function : get_args
    parameters used in .add_argument
    1. metavar - Provide a hint to the user about the data type.
    - By default, all arguments are strings.

    2. type - The actual Python data type
    - (note the lack of quotes around str)

    3. help - A brief description of the parameter for the usage
    """

    parser = argparse.ArgumentParser(
    description='Arguments of main',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('run_with',
    metavar='run_with',
    default='Spikes',
    type=str,
    help='"Both" or "Spikes" or "Image"')

    parser.add_argument('base_data_folder_key',
    metavar='base_data_folder_key',
    default='NS',
    type=str,
    help='"NS" or "Long" or "Sparse"')

    parser.add_argument('data_folder_name',
    metavar='data_folder_name',
    default='data_100KsamplesEvery2Frames_5secslong_halfsizeres',
    type=str,
    help='Name of folder the input data and the results are stored')

    parser.add_argument('n_splits',
    metavar='n_splits',
    default=10,
    type=int,
    help='"number of splits for the TimeSeriesSplit')

    parser.add_argument('starting_iter',
    metavar='starting_iter',
    default=0,
    type=int,
    help='which split to start from')

    parser.add_argument('ending_iter',
    metavar='ending_iter',
    default=10,
    type=int,
    help='which split to end in')

    return parser.parse_args()


def main(run_with='Spikes', base_data_folder_key='NS', data_folder_name='data_100KsamplesEvery2Frames_5secslong_halfsizeres',
         n_splits=10, starting_iter=0, ending_iter=10):
    """

    :param run_with: 'Both' or 'Spikes' or "Image'
    :param base_data_folder_key: 'NS' or 'Long' or 'Sparse'
    :param data_folder_name: Name of folder the input data and the results are stored
    :param n_splits: number of splits for the TimeSeriesSplit
    :param starting_iter: which split to start from
    :param ending_iter: which split to end in
    :return: Nothing
    """

    base_data_folders = {'NS': r'/ceph/scratch/gdimitriadis/Neuroseeker/AK_33.1/2018_04_30-11_38/Analysis/NNs',
                         'Long': r'/ceph/scratch/gdimitriadis/Neuroseeker/AK_33.1/2018_04_30-11_38/Analysis/NeuropixelSimulations/Long/NNs',
                         'Sparse': r'/ceph/scratch/gdimitriadis/Neuroseeker/AK_33.1/2018_04_30-11_38/Analysis/NeuropixelSimulations/Sparce/NNs'}

    base_data_folder = base_data_folders[base_data_folder_key]
    run_with = [run_with]

    data_folder = join(base_data_folder, 'Data', 'TimeSeriesSplit', data_folder_name)
    input_data_name_X = "X_buffer.npy"
    input_data_name_Y = "Y_buffer.npy"

    headers = np.load(join(data_folder, 'binary_headers.npz'), allow_pickle=True)
    X = np.memmap(join(data_folder, input_data_name_X), dtype=headers['dtype'][0], shape=tuple(headers['shape_X']))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    Y = np.memmap(join(data_folder, input_data_name_Y), dtype=headers['dtype'][0], shape=tuple(headers['shape_Y']))
    starting_images = Y[:, 0:1, :, :]
    ending_images = Y[:, 1, :, :]

    print(X.shape)
    print(starting_images.shape)
    print(ending_images.shape)

    frames_used = headers['shape_X'][1]

    tscv = TimeSeriesSplit(gap=frames_used, max_train_size=None, n_splits=n_splits, test_size=None)

    i = 0
    histories_of_losses = []
    histories_of_val_losses = []

    train_indices = []
    test_indices = []

    for train_index, test_index in tscv.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    for i in np.arange(starting_iter, ending_iter):

        randomized_indices = np.random.shuffle(np.arnage(0, train_indices[0][-1]))
        train_index = np.array(train_indices[i])[randomized_indices]
        test_index = np.array(test_indices[i])[randomized_indices]

        print(len(train_index))
        print('TRAIN from {} to {}, TEST from {} to {}'.format(train_index[0], train_index[-1], test_index[0], test_index[-1]))
        start = timeit.timeit()
        X_train = X[train_index]
        X_test = X[test_index]
        starting_images_train, starting_images_test = starting_images[train_index], starting_images[test_index]
        ending_images_train, ending_images_test = ending_images[train_index], ending_images[test_index]

        print('Finished loading data in {}\n'.format(timeit.timeit() - start))

        '''
        batch_size = 500
        gen = generator_random(X, starting_images_train, ending_images_train, train_index, batch_size=batch_size)
        num_of_samples = train_index.shape[0]
        one_extra = 0
        if num_of_samples % batch_size:
            one_extra = 1
        steps_per_epoch = num_of_samples // batch_size + one_extra
        print('STEPS PER EPOCH = '.format(steps_per_epoch))
        '''

        if 'Both' in run_with:
            model_full = build_network(X.shape, ending_images.shape, spikes_images_type='Both')
            print(model_full.summary())

            full_checkpoint_file = (join(data_folder, 'both_latest_model_SSTiter_{}.h5'.format(i)))
            full_checkpoint = ModelCheckpoint(full_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True,
                                               mode='min')
            full_callbacks_list = [full_checkpoint]
            #model_history = model_full.fit_generator(gen, steps_per_epoch=steps_per_epoch,
            #                                         validation_data=([X_test, starting_images_test], ending_images_test),
            #                                         epochs=300, callbacks=full_callbacks_list)
            model_history = model_full.fit([X_train, starting_images_train], ending_images_train,
                                                     validation_data=(
                                                     [X_test, starting_images_test], ending_images_test),
                                                     epochs=300, callbacks=full_callbacks_list)
            model_full.save(join(data_folder, 'both_final_model_SSTiter_{}.h5'.format(i)))

        if 'Spikes' in run_with:
            model_spikes = build_network(X.shape, ending_images.shape, spikes_images_type='Spikes')
            print(model_spikes.summary())

            spikes_checkpoint_file = (join(data_folder, 'spikes_latest_model_SSTiter_{}.h5'.format(i)))
            spikes_checkpoint = ModelCheckpoint(spikes_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            spikes_callbacks_list = [spikes_checkpoint]
            #model_history = model_spikes.fit_generator(gen, steps_per_epoch=steps_per_epoch,
            #                                           validation_data=([X_test, starting_images_test], ending_images_test),
            #                                           epochs=300, callbacks=spikes_callbacks_list)
            model_history = model_spikes.fit([X_train, starting_images_train], ending_images_train,
                                                       validation_data=(
                                                       [X_test, starting_images_test], ending_images_test),
                                                       epochs=300, callbacks=spikes_callbacks_list)
            model_spikes.save(join(data_folder, 'spikes_final_model_SSTiter_{}.h5'.format(i)))

        if 'Image' in run_with:
            model_pictures = build_network(X.shape, ending_images.shape, spikes_images_type='Image')
            print(model_pictures.summary())

            pictures_checkpoint_file = (join(data_folder, 'pictures_latest_model_SSTiter_{}.h5'.format(i)))
            pictures_checkpoint = ModelCheckpoint(pictures_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            pictures_callbacks_list = [pictures_checkpoint]

            #model_history = model_pictures.fit_generator(gen, steps_per_epoch=steps_per_epoch,
            #                                             validation_data=([X_test, starting_images_test], ending_images_test),
            #                                             epochs=300, callbacks=pictures_callbacks_list)
            model_history = model_pictures.fit([X_train, starting_images_train], ending_images_train,
                                                         validation_data=(
                                                         [X_test, starting_images_test], ending_images_test),
                                                         epochs=300, callbacks=pictures_callbacks_list)
            model_pictures.save(join(data_folder, 'pictures_final_model_SSTiter_{}.h5'.format(i)))

        histories_of_losses.append(model_history.history['loss'])
        histories_of_val_losses.append(model_history.history['val_loss'])

        i = i+1

        np.save(join(data_folder, 'loss_histories_of_{}.npy'.format(run_with[0])), np.array(histories_of_losses))
        np.save(join(data_folder, 'val_loss_histories_of_{}.npy'.format(run_with[0])), np.array(histories_of_val_losses))



if __name__ == "__main__":
    args = get_args()
    main(args.run_with, args.base_data_folder_key, args.data_folder_name, args.n_splits, args.starting_iter, args.ending_iter)
