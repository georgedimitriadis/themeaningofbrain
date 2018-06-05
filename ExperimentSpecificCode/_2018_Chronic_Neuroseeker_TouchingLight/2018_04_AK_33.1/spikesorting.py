

import numpy as np
from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as clean
from GUIs.Kilosort import create_data_cubes as c_cubes
from Layouts.Probes.Neuroseeker import probes_neuroseeker as ps


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
                16: r'2018_05_10-15_57'}


number_of_channels_in_binary_file = 1368
base_folder = r'E:\Data\Neuroseeker_chronic_temp'
date_number = 5

kilosort_folder = join(base_folder, date_folders[date_number], 'Analysis', 'Kilosort')

base_info_folder = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker'
prb_file = join(base_info_folder, 'ap_only_prb.txt')

binary_data_filename = join(base_folder, date_folders[date_number], 'Data', 'Amplifier_APs.bin')


# Just once to create the correct prb file with only the AP channels in (1368)
#ps.create_1368channels_neuroseeker_prb(base_info_folder, prb_file)
# ----------------------------------------------------------------
'''
c_cubes.generate_average_over_spikes_per_template_multiprocess(kilosort_folder,
                                                               binary_data_filename,
                                                               number_of_channels_in_binary_file,
                                                               cut_time_points_around_spike=100)
'''

clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_data_filename,
                              prb_file=prb_file,
                              type_of_binary=np.float16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
