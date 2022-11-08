

from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as clean
import numpy as np

base_folder_v = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Vertical\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

base_folder_a = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                r'Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\\' + \
                r'Experiment_2_T18_48_25_And_Experiment_3_T19_41_07\Kilosort'

binary_v = join(base_folder_v, r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin')
binary_a = join(base_folder_a, r'Exp2and3_2017_03_28T18_48_25_Amp_S16_LP3p5KHz_mV.bin')
number_of_channels_in_binary_file = 1440

prb_file = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker\prb.txt'

clean.cleanup_kilosorted_data(base_folder_v,
                              number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_v,
                              prb_file=prb_file,
                              type_of_binary=np.int16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)


clean.cleanup_kilosorted_data(base_folder_a,
                              number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_a,
                              prb_file=prb_file,
                              type_of_binary=np.int16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)


