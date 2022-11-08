

from os.path import join
import numpy as np

base_save_folder = r'D:\\'
rat_folder = r'AK_33.1'


date_folders = {1:r'2018_04_30-11_38'}


BRAIN_REGIONS = {'Cortex MPA': 8100, 'CA1': 6200, 'CA3': 5030,
                 'Thalamus LPMR': 4410, 'Thalamus Po': 3420, 'Thalamus VPM': 2200,
                 'Zona Incerta': 1210, 'Subthalamic Nucleus': 750, 'Internal Capsule': 300}


# Corrections of the number of data points to use because there seems to be a problem with the last frames of the video
BRAIN_DATA_UP_TO_QUARTER_SECOND = 13506
BRAIN_DATA_UP_TO_FRAME = 405180
BRAIN_DATA_UP_TO_POINT = 67259980
