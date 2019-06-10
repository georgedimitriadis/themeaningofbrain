"""
This module carries all the constants for the 33.1 rat.
"""

import os.path as path
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')

NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 1368
NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE = 72
BINARY_FILE_ENCODING = np.int16
SAMPLING_FREQUENCY = 20000

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25


LENGTH_OF_SIX_PIP_SOUND = 14285  # 6 pips at 8.4Hz = 0.71s
LENGTH_OF_ONE_PIP = 500  # 25ms
TIME_BETWEEN_PIPS = 2257  # (14285 - 6*500) / 5
INTER_SOUND_INTERVAL = 4000  # 200ms
