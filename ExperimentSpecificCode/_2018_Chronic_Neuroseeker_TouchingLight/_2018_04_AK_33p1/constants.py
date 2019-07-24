"""
This module carries all the constants for the 33.1 rat.
"""

import os.path as path
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'F:\Neuroseeker chronic'
rat_folder = r'AK_33.1'


date_folders = {1: r'2018_04_23-11_49',
                2: r'2018_04_24-10_12',
                3: r'2018_04_25-11_10',
                4: r'2018_04_26-12_13',
                5: r'2018_04_27-09_44',
                6: r'2018_04_28-19_20',
                7: r'2018_04_29-18_18',
                8: r'2018_04_30-11_38',
                9: r'2018_05_01-11_08',
                10: r'2018_05_02-09_27',
                11: r'2018_05_05-16_44',
                12: r'2018_05_06-14_16',
                13: r'2018_05_07-14_01',
                14: r'2018_05_08-13_12',
                15: r'2018_05_09-12_26',
                16: r'2018_05_10-15_57',
                17: r'2018_05_14-11_59'}


NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 1368
NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE = 72
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158
SAMPLING_FREQUENCY = 20000

NUMBER_OF_IMFS = 13

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

PIXEL_PER_FRAME_TO_CM_PER_SECOND = 22.73
PIXEL_PER_FRAME_TO_METERS_PER_SECOND = PIXEL_PER_FRAME_TO_CM_PER_SECOND / 100
PIXEL_PER_FRAME_TO_KM_PER_HOUR = PIXEL_PER_FRAME_TO_METERS_PER_SECOND * 3.6

BRAIN_REGIONS = {'Cortex MPA': 8100, 'CA1': 6200, 'CA3': 5030,
                 'Thalamus LPMR': 4410, 'Thalamus Po': 3420, 'Thalamus VPM': 2200,
                 'Zona Incerta': 1210, 'Subthalamic Nucleus': 750, 'Internal Capsule': 300}

# Corrections of the number of data points to use because there seems to be a problem with the last frames of the video
BRAIN_DATA_UP_TO_QUARTER_SECOND = 13506
BRAIN_DATA_UP_TO_FRAME = 405180
BRAIN_DATA_UP_TO_POINT = 67259980
