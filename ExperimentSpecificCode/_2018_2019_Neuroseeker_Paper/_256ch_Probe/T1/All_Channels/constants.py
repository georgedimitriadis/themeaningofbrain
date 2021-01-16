

from os.path import join
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes'
prb_file = join(probe_layout_folder, 'probe_imec_256channels_file.txt')

data_filename = r'amplifier2017-02-08T20_54_26.bin'
base_save_folder = r'D:\_256channels'
cell_folder = r'T1'
decimation_type_folder = r'Full_Channels'

bad_channels =               [232, 162, 8,  28, 32, 30,  131, 148, 151, 137, 171, 149, 34, 19, 94, 7]
neighbours_to_bad_channels = [236, 212, 12, 24, 63, 221, 191, 143, 183, 166, 170, 144, 62, 23, 38, 4]

NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 256
BINARY_FILE_ENCODING = np.int16
SAMPLING_FREQUENCY = 20000


PROBE_DIMENSIONS = [100, 102]
POSITION_MULT = 6