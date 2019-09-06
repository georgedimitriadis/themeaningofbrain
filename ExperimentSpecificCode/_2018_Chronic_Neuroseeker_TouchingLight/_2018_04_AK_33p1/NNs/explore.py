

import cv2
import numpy as np
import pickle

try:
    from . import constants as const
except ImportError:
    from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1.NNs \
    import constants as const


spiky_model = pickle.load(open(const.spiky_model_file, 'rb'))
picture_model = pickle.load(open(const.picture_model_file, 'rb'))

data = np.load(const.save_data_file)