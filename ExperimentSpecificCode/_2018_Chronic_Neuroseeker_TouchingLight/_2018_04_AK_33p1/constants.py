
import os.path as path
import numpy as np

probe_layout_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = path.join(probe_layout_folder, 'ap_only_prb.txt')


base_save_folder = r'D:\Data\George'
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

start_end_ephys_points_of_video_recording = {8: [147845, 72448825]}


NUMBER_OF_CHANNELS_IN_BINARY_FILE = 1368
BINARY_FILE_ENCODING = np.int16
CAMERA_TTL_PULSES_TIMEPOINT_PERIOD = 158 #158

PROBE_DIMENSIONS = [100, 8100]
POSITION_MULT = 2.25

#BRAIN_REGIONS = {'Cortex MPA': 8070, 'Corpus Calosum': 6600, 'CA1': 6210, 'CA3': 4950,
#                 'Thalamus LPMR': 4440, 'Thalamus Po': 3550, 'Thalamus VPM': 1950,
#                 'Zona Incerta': 1250, 'Subthalamic Nucleus': 750, 'Internal Capsule': 300}

BRAIN_REGIONS = {'Cortex MPA': 8070, 'CA1': 6400, 'CA3': 4950,
                 'Thalamus LPMR': 4440, 'Thalamus Po': 3550, 'Thalamus VPM': 1950,
                 'Zona Incerta': 1250, 'Subthalamic Nucleus': 750, 'Internal Capsule': 300}