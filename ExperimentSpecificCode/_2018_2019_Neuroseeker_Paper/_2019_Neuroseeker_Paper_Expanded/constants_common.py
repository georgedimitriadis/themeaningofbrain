

from os.path import join
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = join(probe_layout_folder, 'ap_only_prb.txt')


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
