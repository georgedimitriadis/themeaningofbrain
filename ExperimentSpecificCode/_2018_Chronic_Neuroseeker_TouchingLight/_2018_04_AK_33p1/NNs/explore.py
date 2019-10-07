

import cv2
import numpy as np
import pickle
import keras

'''
try:
    from . import constants as const
except ImportError:
    from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1.NNs \
    import constants as const
'''
from constants import *

spiky_model = keras.models.load_model(spiky_model_file)
picture_model = keras.models.load_model(picture_model_file)

data = np.load(save_data_file)


X_test = data['X_test']
starting_images_test = data['starting_images_test']
ending_images_test = data['ending_images_test']

X_test_1 = X_test[1, :, :].reshape(1, X_test.shape[1], X_test.shape[2])
s_i_test_1 = starting_images_test[1, :, :, :].reshape(1, 1, starting_images_test.shape[2], starting_images_test.shape[3])
e_i_test_1 = ending_images_test[1, :, :].reshape(ending_images_test.shape[1], ending_images_test.shape[2])

im_pred_pic = picture_model.predict(s_i_test_1)

im_pred_spikes = spiky_model.predict([X_test_1, s_i_test_1])







index = 1000
spikes = X[index, :, :, :].reshape(X.shape[1], X.shape[2], X.shape[3])
si = starting_images[index, :, :,:].reshape(1, 1, starting_images.shape[2], starting_images.shape[3])
ei = np.squeeze(ending_images[index, :, :])
pred_spikes = picture_model.predict(s_1)
pred_spikes = spiky_model.predict([X_1, s_1])
pred_pics = picture_model.predict(s_1)

