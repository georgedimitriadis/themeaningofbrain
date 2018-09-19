
import os.path as path
import numpy as np


# probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
probe_layout_folder = r'E:\Data\Neuroseeker_chronic_temp\AK_34.4\2018_06_18-12_50\Analysis\Kilosort'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')

# base_save_folder = r'D:\Data\George'
base_save_folder = r'E:\Data\Neuroseeker_chronic_temp'
rat_folder = r'AK_34.4'


date_folders = {1: r'2018_06_18-12_50'}


NUMBER_OF_CHANNELS_IN_BINARY_FILE = 1368
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158

