

from os.path import join
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes'
prb_file = join(probe_layout_folder, 'probe_imec_256channels_file.txt')

data_filename = r'amplifier2017-02-23T18_25_19_int16.bin'
base_save_folder = r'D:\_256channels'
cell_folder = r'CoP1'
decimation_type_folder = r'Full_Channels'

bad_channels =               [232, 162, 8,  28, 32, 30,  131, 148, 151, 137, 171, 149, 34, 19, 94, 7]#,
                              #35,  37,  26,  9,  10, 22, 40, 39,
                              #45, 48, 11,  49,  43,
                              #46, 3,  14, 18, 125, 164, 90,  119, 129, 132, 179, 152, 180, 177, 52]
neighbours_to_bad_channels = [236, 212, 12, 24, 63, 221, 191, 143, 183, 166, 170, 144, 62, 23, 38, 4]#,
                              #225, 223, 125, 17, 14, 44, 95, 21,
                              #20, 44, 111, 106, 77,
                              #47, 13, 13, 50, 120, 167, 123, 65,  128, 165, 153, 147, 146, 176, 110]

NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 256
BINARY_FILE_ENCODING = np.int16
SAMPLING_FREQUENCY = 20000


PROBE_DIMENSIONS = [100, 102]
POSITION_MULT = 6