
import numpy as np

base_save_folder = r'D:\\'
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

bad_channels = np.array([np.arange(684, 727, 1), np.arange(1140, 1176, 1)])


BRAIN_REGIONS = {'Cortex MPA': 7220, 'CA1': 5460, 'CA3': 4475,
                 'Thalamus LDVL': 3735, 'Thalamus Po': 2730, 'Thalamus VPM': 2400,
                 'Zona Incerta': 720}

BRAIN_REGIONS = {'Cortex MPA': 7990, 'CA1': 5460, 'CA3': 4475,
                 'Thalamus LDVL': 3735, 'Thalamus Po': 2730, 'Thalamus VPM': 2400,
                 'Zona Incerta': 720}