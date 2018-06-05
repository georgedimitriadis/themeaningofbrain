

import numpy as np
from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as clean
from GUIs.Kilosort import create_data_cubes as c_cubes
from Layouts.Probes.Neuroseeker import probes_neuroseeker as ps
from . import constants as const
import spikesorting_tsne as tsne


# This is to create the correct prb file with only the AP channels in (1368). It should never be run again
ps.create_1368channels_neuroseeker_prb(const.probe_layout_folder, const.prb_file)
# ----------------------------------------------------------------


# CLEANING THE KILOSORT RESULTS ----------------------------------
kilosort_folder = join(r'F:\Neuroseeker\AK_33.1', const.date_folders[1], 'Analysis', 'Kilosort')
binary_data_filename = join(r'F:\Neuroseeker\AK_33.1', const.date_folders[1], 'Data', 'Amplifier_APs.bin')

'''
# Create once the data cube of the average template spike
c_cubes.generate_average_over_spikes_per_template_multiprocess(kilosort_folder,
                                                               binary_data_filename,
                                                               const.NUMBER_OF_CHANNELS_IN_BINARY_FILE,
                                                               cut_time_points_around_spike=100)
'''

# Run the GUI that helps clean the templates
clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=const.NUMBER_OF_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const.prb_file,
                              type_of_binary=np.float16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
# ----------------------------------------------------------------


# T-SNE THE CLEANED SPIKES ---------------------------------------
