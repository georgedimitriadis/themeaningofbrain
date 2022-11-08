"""
This module carries all the constants for the 47.2 rat.
"""

import os.path as path
import numpy as np

probe_layout_folder = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'D:\\'
#base_save_folder = r'Y:\swc\kampff\George\DataAndResults\Experiments\Awake\NeuroSeeker'
rat_folder = r'AK_47.2'


date_folders = {1: r'2019_06_18-10_15',
                2: r'2019_06_19-11_00',
                3: r'2019_06_20-12_02',
                4: r'2019_06_21-10_34',
                5: r'2019_06_22-12_14',
                6: r'2019_06_23-20_37',
                7: r'2019_06_24-12_19',
                8: r'2019_06_25-12_50',
                21: r'2019_07_08-11_15'}

bad_channels = np.array([np.arange(684, 727, 1), np.arange(1140, 1176, 1)], dtype=object)


NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 1368
NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE = 72
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158
SAMPLING_FREQUENCY = 20000
LFP_DOWNSAMPLE_FACTOR = 4

NUMBER_OF_IMFS = 13

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

PIXEL_PER_FRAME_TO_CM_PER_SECOND = 22.73
PIXEL_PER_FRAME_TO_METERS_PER_SECOND = PIXEL_PER_FRAME_TO_CM_PER_SECOND / 100
PIXEL_PER_FRAME_TO_KM_PER_HOUR = PIXEL_PER_FRAME_TO_METERS_PER_SECOND * 3.6

BRAIN_REGIONS = {'Cortex MPA': 7220, 'CA1': 5460, 'CA3': 4475,
                 'Thalamus LDVL': 3735, 'Thalamus Po': 2730, 'Thalamus VPM': 2400,
                 'Zona Incerta': 720}

BRAIN_REGIONS = {'Cortex MPA': 7990, 'CA1': 5460, 'CA3': 4475,
                 'Thalamus LDVL': 3735, 'Thalamus Po': 2730, 'Thalamus VPM': 2400,
                 'Zona Incerta': 720}

BRAIN_REGIONS = {'Cortex MPA': 7990, 'CA1': 5460, 'CA3': 4475,
                 'Thalamus LDVL': 3735, 'Thalamus Po': 2730, 'Thalamus VPM': 2400,
                 'Zona Incerta': 1100}


