
import os.path as path
import numpy as np


# probe_layout_folder = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker'


# base_save_folder = r'D:\Data\George'
base_save_folder = r'E:\Data\Neuroseeker_chronic'
rat_folder = r'AK_34.4'


date_folders = {1: r'2018_06_18-12_50',
                2:'2018_07_06-14_04'}

probe_layout_folder = path.join(base_save_folder, rat_folder, date_folders[2], 'Analysis', 'Kilosort')
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')

NUMBER_OF_CHANNELS_IN_BINARY_FILE = 1368
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158


PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

BRAIN_REGIONS = {'Cortex LPA': 8100, 'Corpus Calosum': 6240, 'CA1': 5780, 'CA2': 5190, 'CA3': 4870,
                 'Thalamus LPLR': 4250, 'Thalamus Po': 3300, 'Thalamus VPM': 2660,
                 'Zona Incerta': 1180, 'Subthalamic Nuclei': 200}
