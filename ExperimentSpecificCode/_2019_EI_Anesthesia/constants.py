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
