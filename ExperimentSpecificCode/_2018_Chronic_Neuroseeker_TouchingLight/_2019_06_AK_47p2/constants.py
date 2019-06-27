"""
This module carries all the constants for the 33.1 rat.
"""

import os.path as path
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'F:\Neuroseeker chronic'
rat_folder = r'AK_47.2'


date_folders = {1: r'2019_06_18-10_15',
                2: r'2019_06_19-11_00',
                3: r'2019_06_20-12_02',
                4: r'2019_06_21-10_34',
                5: r'2019_06_22-12_14',
                6: r'2019_06_23-20_37',
                7: r'2019_06_24-12_19',
                8: r'2019_06_25-12_50'}


NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 1368
NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE = 72
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158
SAMPLING_FREQUENCY = 20000

NUMBER_OF_IMFS = 13

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

PIXEL_PER_FRAME_TO_METERS_PER_SECOND = 0.1875
PIXEL_PER_FRAME_TO_KM_PER_HOUR = PIXEL_PER_FRAME_TO_METERS_PER_SECOND * 3.6

BRAIN_REGIONS = {'Cortex MPA': 8100, 'CA1': 6200, 'CA3': 5030,
                 'Thalamus LPMR': 4410, 'Thalamus Po': 3420, 'Thalamus VPM': 2200,
                 'Zona Incerta': 1210, 'Subthalamic Nucleus': 750, 'Internal Capsule': 300}

# Corrections of the number of data points to use because there seems to be a problem with the last frames of the video
BRAIN_DATA_UP_TO_QUARTER_SECOND = 13506
BRAIN_DATA_UP_TO_FRAME = 405180
BRAIN_DATA_UP_TO_POINT = 67259980
