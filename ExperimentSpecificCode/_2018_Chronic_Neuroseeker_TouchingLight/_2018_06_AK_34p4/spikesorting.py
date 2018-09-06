

import numpy as np
import pandas as pd
from os.path import join
#from GUIs.Kilosort import clean_kilosort_templates as clean
from spikesorting_tsne_guis import clean_kilosort_templates as clean
from GUIs.Kilosort import create_data_cubes as c_cubes
from Layouts.Probes.Neuroseeker import probes_neuroseeker as ps
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_06_AK_34p4 import constants as const
from spikesorting_tsne import tsne, visualization as viz, preprocessing_kilosort_results as preproc_kilo, \
     io_with_cpp as tsne_io


# This is to create the correct prb file with only the AP channels in (1368). It should never be run again
# ps.create_1368channels_neuroseeker_prb(const.probe_layout_folder, const.prb_file)
# ----------------------------------------------------------------

# FOLDERS NAMES --------------------------------------------------
date = 1
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne')
# ----------------------------------------------------------------


# CLEANING THE KILOSORT RESULTS ----------------------------------

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
                              type_of_binary=const.BINARY_FILE_ENCODING,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
# ----------------------------------------------------------------


# T-SNE THE CLEANED SPIKES ---------------------------------------

# Find how many spikes are clean (demarcated as not noise)
templates_of_spikes = np.load(join(kilosort_folder, 'spike_templates.npy'))
number_of_raw_spikes = len(templates_of_spikes)
template_markings = preproc_kilo.get_template_marking(kilosort_folder)
clean_templates = np.argwhere(template_markings > 0)
number_of_clean_templates = len(clean_templates)
clean_spikes = np.argwhere(np.in1d(templates_of_spikes, clean_templates) > 0)
number_of_clean_spikes = len(clean_spikes)


# Get a representative sample of spikes for t-sne


representative_indices, small_templates, large_templates = \
    preproc_kilo.find_spike_indices_for_representative_tsne(kilosort_folder, tsne_folder, 5000, 700000)

template_features_matrix = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder, tsne_folder,
                                                                                    spikes_used_with_original_indexing=
                                                                                    representative_indices)

template_features_matrix = np.load(join(tsne_folder, 'data_to_tsne_(699959, 645).npy'))


# First run of t-sne
num_dims = 2
perplexity = 100
theta = 0.3
iterations = 4000
random_seed = 1
verbose = 3

tsne_results = tsne.t_sne(template_features_matrix, files_dir=tsne_folder, num_dims=num_dims, perplexity=perplexity,
                          theta=theta, iterations=iterations, random_seed=random_seed, verbose=verbose)
spike_info = preproc_kilo.generate_spike_info(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder)


# Run t-sne again with different parameters starting from the already calculated hd distances
num_dims = 2
perplexity = 100
theta = 0.3
iterations = 4000
random_seed = 1
verbose = 3

tsne_results = tsne.t_sne_from_existing_distances(files_dir=tsne_folder, data_has_exageration=True, num_dims=num_dims,
                                                  theta=theta, iterations=iterations, random_seed=random_seed,
                                                  verbose=verbose)
spike_info = preproc_kilo.generate_spike_info(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder)


# OR Load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_folder)

# and previously generated spike_info
spike_info = pd.read_pickle(join(tsne_folder, 'spike_info.df'))


# Have a look
viz.plot_tsne_of_spikes(spike_info=spike_info, legent_on=False)


# ------------------------------------------
# Now run the T-Sne Gui to manual sort
# ------------------------------------------


