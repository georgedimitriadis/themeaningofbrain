
"""
This is the original cleaning and t-sne embedding of all the template (using a subset of spikes) on the kilosorted
data of 2018_04_30-11_38.

It also has code that selects the largest MUA templates and t-sne each one of them separately. It does using the
kilosort template distances. This doesn't work at all because each template has only one or two distance to templates that
are not noise. The testing_pca_for_tsne.py has code that t-snes the MUA templates using PCs).
"""

import os
import numpy as np
import pandas as pd
from os.path import join
from GUIs.Kilosort import clean_kilosort_templates as clean
from GUIs.Kilosort import create_data_cubes as c_cubes
from Layouts.Probes.Neuroseeker import probes_neuroseeker as ps
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from spikesorting_tsne import tsne, visualization as viz, preprocessing_kilosort_results as preproc_kilo, \
     io_with_cpp as tsne_io, positions_on_probe as sp_pos, constants as ct
from spikesorting_tsne_guis import spikesort_on_tsne

# This is to create the correct prb file with only the AP channels in (1368). It should never be run again
# ps.create_1368channels_neuroseeker_prb(const.probe_layout_folder, const.prb_file)
# ----------------------------------------------------------------

# FOLDERS NAMES --------------------------------------------------
date = 8
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne')
barnes_hut_exe_dir=r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'
# ----------------------------------------------------------------


# CLEANING THE KILOSORT RESULTS ----------------------------------

'''
# Create once the data cube of the average template spike
c_cubes.generate_average_over_spikes_per_template_multiprocess(kilosort_folder,
                                                               binary_data_filename,
                                                               const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                                                               cut_time_points_around_spike=100)


# Run the GUI that helps clean the templates
clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const.prb_file,
                              type_of_binary=np.int16,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
# ----------------------------------------------------------------
'''


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
spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder)


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
spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder)


# OR Load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_folder)

# and previously generated spike_info
spike_info = pd.read_pickle(join(tsne_folder, 'spike_info.df'))


# Have a look
viz.plot_tsne_of_spikes(spike_info=spike_info, legent_on=False)


# ------------------------------------------
# Now run the T-Sne Gui to manual sort
# ------------------------------------------


# ------------------------------------------
# Get the positions of the templates and have a look
sp_pos.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS, const.POSITION_MULT)


# -----------------------------------------
# -----------------------------------------
# CREATE T-SNE FOR EACH MUA TEMPLATE WITH MORE THAN 10000 SPIKES INDIVIDUALLY
# Find the mua templates with large spike count

mua_templates = np.squeeze(np.argwhere(template_markings == 4))
# mua_spikes = np.squeeze(np.argwhere(np.in1d(templates_of_spikes, mua_templates) > 0))

size_of_mua_templates = []
large_mua_templates = []
number_of_spikes_in_large_mua_templates = 10000
for mua_template in mua_templates:
    mua_spikes = np.argwhere(np.in1d(templates_of_spikes, mua_template) > 0)
    size_of_mua_templates.append(len(mua_spikes))
    if len(mua_spikes) >= number_of_spikes_in_large_mua_templates:
        large_mua_templates.append(mua_template)


# -------------------------------------
# MANUALLY PICK A TEMPLATE AND T-SNE IT
# (remember to create a directory called template_xx where xx is the number of the template)
large_mua_template = 2
mua_template = large_mua_templates[large_mua_template]

tsne_folder_single_template = join(tsne_folder, 'template_{}'.format(mua_template))


mua_spikes = np.argwhere(np.in1d(templates_of_spikes, mua_template) > 0)
np.save(join(tsne_folder_single_template, ct.INDICES_OF_SPIKES_USED_FILENAME), mua_spikes)

template_features_matrix = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder, tsne_folder_single_template,
                                                                                    spikes_used_with_original_indexing=
                                                                                    mua_spikes)

# First run of t-sne
num_dims = 2
perplexity = 100
theta = 0.2
iterations = 2000
random_seed = 1
verbose = 3

tsne_results = tsne.t_sne(template_features_matrix, files_dir=tsne_folder_single_template, exe_dir=barnes_hut_exe_dir,
                          num_dims=num_dims, perplexity=perplexity,
                          theta=theta, iterations=iterations, random_seed=random_seed, verbose=verbose)

# OR run only the Barnes Hut if the distances have already been calculated (i.e. there is a data.dat file in the
# t-sne directory generated by running the tsne.t_sne function)
tsne_results = tsne.t_sne_from_existing_distances(files_dir=tsne_folder_single_template, data_has_exageration=True,
                                                  exe_dir=barnes_hut_exe_dir, num_dims=2, theta=0.2, eta=200,
                                                  exageration=12.0, iterations=4000, random_seed=1, verbose=2)

# OR Load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_folder_single_template)

# Get the spike info (if not already there)
spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder, tsne_folder=tsne_folder_single_template)
# OR Load
spike_info = pd.read_pickle(join(tsne_folder_single_template, 'spike_info.df'))

# Call up the gui to have a look and sort manually)
spikesort_on_tsne.spikesort_gui(load_previous_dataset=True)


# -------------------------------------
# OR DO ALL THE ABOVE IN A LOOP FOR ALL LARGE MUA TEMPLATES
for mua_template in large_mua_templates[2:]:
    tsne_folder_single_template = join(tsne_folder, 'template_{}'.format(mua_template))
    os.mkdir(tsne_folder_single_template)

    mua_spikes = np.argwhere(np.in1d(templates_of_spikes, mua_template) > 0)
    np.save(join(tsne_folder_single_template, ct.INDICES_OF_SPIKES_USED_FILENAME), mua_spikes)

    template_features_matrix = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder,
                                                                                        tsne_folder_single_template,
                                                                                        spikes_used_with_original_indexing=
                                                                                        mua_spikes)
    num_dims = 2
    perplexity = 100
    theta = 0.2
    iterations = 3000
    random_seed = 1
    verbose = 3

    tsne_results = tsne.t_sne(template_features_matrix, files_dir=tsne_folder_single_template,
                              exe_dir=barnes_hut_exe_dir,
                              num_dims=num_dims, perplexity=perplexity,
                              theta=theta, iterations=iterations, random_seed=random_seed, verbose=verbose)
    spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder,
                                                                 tsne_folder=tsne_folder_single_template)

