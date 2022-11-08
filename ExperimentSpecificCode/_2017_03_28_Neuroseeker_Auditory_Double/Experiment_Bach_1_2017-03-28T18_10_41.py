
# Clean up

from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as clean
from GUIs.Kilosort import create_data_cubes as c_cubes
import numpy as np

data_folder_a = r'D:\Data\George\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Data\Experiment_Bach_1_2017-03-28T18_10_41'
kilosort_folder_a = r'D:\Data\George\Neuroseeker_2017_03_28_Anesthesia_Auditory_DoubleProbes\Angled\Analysis\Experiment_Bach_1_2017-03-28T18_10_41\Kilosort'

binary_a = join(data_folder_a, r'Amplifier_APs.bin')
number_of_channels_in_binary_file = 1440

prb_file = r'E:\Code\Mine\themeaningofbrain\Layouts\Probes\Neuroseeker\prb.txt'


# Create once the data cube of the average template spike
c_cubes.generate_average_over_spikes_per_template_multiprocess(kilosort_folder_a,
                                                               binary_a,
                                                               number_of_channels_in_binary_file,
                                                               cut_time_points_around_spike=100)



clean.cleanup_kilosorted_data(kilosort_folder_a,
                              number_of_channels_in_binary_file=number_of_channels_in_binary_file,
                              binary_data_filename=binary_a,
                              prb_file=prb_file,
                              type_of_binary=np.int16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)




avg = np.load(join(kilosort_folder_a, 'avg_spike_template.npy'))

import matplotlib.pyplot as plt
plt.plot(avg[632,:,:].T)