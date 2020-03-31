from Initial_George_Code.constants import *

# from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1.NNs.constants import *


from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, concatenate, Reshape, Flatten, Dropout, \
    CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_spikes_only_network(spike_shape, image_shape):
    input = Input(shape=(spike_shape[1], spike_shape[2]))
    x = CuDNNLSTM(64)(input)
    predictions = Dense(image_shape[1] * image_shape[2], activation='sigmoid')(x)

    predictions = Reshape((image_shape[1], image_shape[2]))(predictions)

    model = Model(inputs=[input], outputs=predictions)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='mse')

    return model


def build_network(spike_shape, image_shape, with_spikes=True):

    # This returns a tensor
    input_0 = Input(shape=(spike_shape[1], spike_shape[2]))
    input_1 = Input(shape=(1, image_shape[1], image_shape[2]))

    ## let's start with 2D convolutions 3 X 3
    x_spikes = CuDNNLSTM(64)(input_0)
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
    model.compile(optimizer=Adam(lr=0.001),
                  loss='mse')

    #print(model.summary())

    return model


data = np.load(save_data_file)
X_train = data['X_train']
X_test = data['X_test']
starting_images_train = data['starting_images_train']
starting_images_test = data['starting_images_test']
ending_images_train = data['ending_images_train']
ending_images_test = data['ending_images_test']

run = [0]

# Run spikes only network
if 0 in run:
    spiky_only_model = build_spikes_only_network(X_train.shape, ending_images_train.shape)
    print(spiky_only_model.summary())

    spiky_only_checkpoint = ModelCheckpoint(spiky_only_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False,
                                       mode='auto')
    spiky_only_callbacks_list = [spiky_only_checkpoint]
    spiky_only_model.fit([X_train, starting_images_train], ending_images_train,
                    validation_data=([X_test, starting_images_test], ending_images_test),
                    epochs=300, callbacks=spiky_only_callbacks_list)

    spiky_only_model.save(spiky_only_model_file)

# Run spikes and image network
if 1 in run:
    spiky_model = build_network(X_train.shape, ending_images_train.shape, with_spikes=True)
    print(spiky_model.summary())

    spiky_checkpoint = ModelCheckpoint(spiky_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
    spiky_callbacks_list = [spiky_checkpoint]
    spiky_model.fit([X_train, starting_images_train], ending_images_train,
              validation_data=([X_test, starting_images_test], ending_images_test),
              epochs=300, callbacks=spiky_callbacks_list)

    spiky_model.save(spiky_model_file)


# Run image only network
if 2 in run:
    picture_model = build_network(X_train.shape, ending_images_train.shape, with_spikes=False)
    print(picture_model.summary())

    picture_checkpoint = ModelCheckpoint(picture_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
    picture_callbacks_list = [picture_checkpoint]
    picture_model.fit([X_train, starting_images_train], ending_images_train,
              validation_data=([X_test, starting_images_test], ending_images_test),
              epochs=300, callbacks=picture_callbacks_list)

    picture_model.save(picture_model_file)




