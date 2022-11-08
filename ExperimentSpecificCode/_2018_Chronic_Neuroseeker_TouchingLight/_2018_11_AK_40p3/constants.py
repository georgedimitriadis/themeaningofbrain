
import os.path as path
import numpy as np


probe_layout_folder = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'D:\\'


experiment_folders = {1: r'AK_40.3_AK_40.4\2018_11_22-10_31',
                      2: r'AK_40.3\2018_12_05-16_18',
                      3: r'AK_40.3\2018_12_11-11_29'}

NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE = 1368
NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE = 72
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

BRAIN_REGIONS = {'Cortex LPA': 8100, 'Corpus Calosum': 6240, 'CA1': 5780, 'CA2': 5190, 'CA3': 4870,
                 'Thalamus LPLR': 4250, 'Thalamus Po': 3300, 'Thalamus VPM': 2660,
                 'Zona Incerta': 1180, 'Subthalamic Nuclei': 200}
