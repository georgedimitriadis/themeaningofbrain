
"""
The pipeline for spikesorting this dataset
"""

import numpy as np
import pandas as pd
from os.path import join

from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, io_with_cpp as tsne_io, tsne as tsne, \
    visualization as viz
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from spikesorting_tsne_guis import clean_kilosort_templates as clean, spikesort_on_tsne

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import BrainDataAnalysis.Spike_Sorting.dense_probe_desnoising as denoise

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const


import sequence_viewer as seq_v


# ----------------------------------------------------------------------------------------------------------------------
# FOLDERS NAMES
date = 8
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis')
data_filename = join(analysis_folder, 'Data', 'Amplifier_APs_Denoised.bin')
kilosort_folder = join(analysis_folder, 'Kilosort')

tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'
tsne_cortex_folder = join(analysis_folder, 'Tsne', 'Cortex')

sampling_freq = const.SAMPLING_FREQUENCY
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 1: DENOISE THE DATA
# CAREFUL: Running this will overwrite any previous file with the same filename !!!!
denoise.denoise_data(binary_data_filename, const.BINARY_FILE_ENCODING, data_filename,
                     const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE, sampling_freq, compute_window_in_secs=1,
                     use_dask_for_parallel=True)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#  STEP 2: HAVE A LOOK AT THE DENOISED DATA

raw_data_denoised = ns_funcs.load_binary_amplifier_data(data_filename,
                                                        number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

# Cut the data in chunks of channels in the same brain region (to allow them to fit in the graphs)
data_denoised_cor = raw_data_denoised[1050:, :]
data_denoised_hyp = raw_data_denoised[850:1050, :]
data_denoised_th = raw_data_denoised[370:850, :]
data_denoised_sth = raw_data_denoised[:370, :]


def space_denoised_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (200*i) for i in np.arange(dat.shape[0])]))
    return result


buffer = 500
pointer = 39537363


seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_cor', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_hyp', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_th', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_sth', transform_name='space_denoised_data')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#  STEP 3: RUN KILOSORT ON THE DENOISED DATA
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 4: CLEAN DENOISED SPIKESORT (RIGHT AFTER KILOSORT)

# a) Create average of templates:
# To create averages of templates use cmd (because the create_data_cubes doesn't work when called from a REPL):
# Go to where the create_data_cubes.py is (in spikesort_tsne_guis/spikesort_tsen_guis) and run the following python command
# (you can use either the raw or the denoised data to create the average)
# python E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis\create_data_cubes.py
#                                                                                original
#                                                                                "F:\Neuroseeker chronic\AK_47.2\2019_06_18-10_15\Analysis\Kilosort"
#                                                                                "F:\Neuroseeker chronic\AK_47.2\2019_06_18-10_15\Data\Amplifier_APs.bin"
#                                                                                1368
#                                                                                50
# (Use single space between parameters, not Enter like here)
# (Change the folders as appropriate for where the data is)

# b) Clean:
clean.cleanup_kilosorted_data(kilosort_folder,
                              number_of_channels_in_binary_file=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const.prb_file,
                              type_of_binary=const.BINARY_FILE_ENCODING,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 5: CREATE TEMPLATE INFO OF ALL THE CLEANED TEMPLATES

# a) Create the positions of the templates on the probe (and have a look)
spp.generate_probe_positions_of_templates(kilosort_folder)
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT)

# b) Create the template_info.df dataframe (or load it if you already have it)
# template_info = preproc_kilo.generate_template_info_after_cleaning(kilosort_folder_denoised, sampling_freq)
template_info = np.load(join(kilosort_folder, 'template_info.df'))
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 6: PICK THE CORTICAL SPIKES AND T-SNE THEM

# a) Pick the cortical spikes
templates_in_cortex = template_info[template_info['position Y'] * const.POSITION_MULT > const.BRAIN_REGIONS['CA1']]
templates_in_cortex.index = np.arange(len(templates_in_cortex)) # This is required to fix the index of the new dataframe

spikes_in_cortex = np.array([])
for t in templates_in_cortex['spikes in template']:
    spikes_in_cortex = np.concatenate((spikes_in_cortex, t))


# b) Make the distance of each spike to its template's closest templates as the feature matrix of T-sne
tsne_cortex_folder = join(analysis_folder, 'Tsne', 'Cortex')

np.save(join(tsne_cortex_folder, 'indices_of_spikes_used.npy'), spikes_in_cortex)  # Very important to save this file!
template_features = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder,
                                                                             tsne_cortex_folder,
                                                                             spikes_used_with_original_indexing=spikes_in_cortex)

# c) Run the T-sne
num_dims = 2
perplexity = 100
theta = 0.3
iterations = 4000
random_seed = 1
verbose = 2

# This is a full T-sne (calculates distances in GPU and the T-sne embedding in CPU)
tsne_results = tsne.t_sne(template_features, files_dir=tsne_cortex_folder, exe_dir=tsne_exe_dir, num_dims=num_dims,
                          perplexity=perplexity, theta=theta, iterations=iterations, random_seed=random_seed,
                          verbose=verbose)

# OR if you have the distances already calculated then run the following (runs only the CPU embedding part)
tsne_results = tsne.t_sne_from_existing_distances(files_dir=tsne_cortex_folder, data_has_exageration=True,
                                                  num_dims=num_dims, theta=theta, iterations=iterations,
                                                  random_seed=random_seed, verbose=verbose, exe_dir=tsne_exe_dir)

# OR load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_cortex_folder)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 7: HAVE A LOOK AT THE T-SNE DATA

# a) Make a spike_info from the t-sne data (will be used later in the t-sne sorting gui)
spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder,
                                                             tsne_folder=tsne_cortex_folder)

# OR load a previously generated spike_info
spike_info = pd.read_pickle(join(tsne_cortex_folder, 'spike_info.df'))

# and have a look
viz.plot_tsne_of_spikes(spike_info=spike_info, legent_on=False)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 8: RUN THE T-SNE SPIKESORTING GUI ON THE ABOVE SPIKE INFO TO REARRANGE SPIKES

# The following will load an empty GUI that you need to load the data (the above spike_info) into it (File/Load Data)
spikesort_on_tsne.spikesort_gui(False)
# You can do the above also from the command line (go where the spikesorting_tsne_guis.spikesort_on_tsne.py is and run
# python spikesort_on_tsne.py False (or True if you have already loaded data before and you want to continue from where
# you left off)

# Every time you save the GUI's state (File/Save) it will put everything you have done in the spike_info.df file it has
# been using (so maybe make a copy of the original one before you start changing things).
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 9: MERGE THE NEW SPIKE_INFO.DF IN THE T-SNE FOLDER (THE RESULT OF THE MANUAL SORTING) WITH THE ORIGINAL
# SPIKE_INFO.DF IN THE KILOSORT FOLDER (MADE FROM THE RESULTS OF THE CLEANED KILOSORT)

# a) Make the spike info from the initial, cleaned, kilosort results
spike_info_after_cleaning = preproc_kilo.generate_spike_info_after_cleaning(kilosort_folder)

# b) Get the spike info form the T-Sne and merge it to the original one
spike_info_cortex_sorted = spike_info
tsne_filename = join(tsne_cortex_folder, 'result.dat')
spike_info_after_cortex_sorting_file = join(kilosort_folder, 'spike_info_after_cortex_sorting.df')
spike_info_after_sorting = preproc_kilo.add_sorting_info_to_spike_info(spike_info_after_cleaning,
                                                                       spike_info_cortex_sorted,
                                                                       tsne_filename=tsne_filename,
                                                                       save_to_file=spike_info_after_cortex_sorting_file)

# OR load it if you have already merged it
spike_info_after_sorting = np.load(spike_info_after_cortex_sorting_file, allow_pickle=True)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# STEP 10: MAKE NEW TEMPLATE_INFO FROM MERGED SPIKE_INFO, RECREATE THE AVERAGE SPIKES OF EACH TEMPLATE DATA CUBE AND
# RECREATE THE NEW TEMPLATE POSITIONS ON THE PROBE

# a) Create the new template_info.df from the updated spike_info.df
template_info = preproc_kilo.generate_template_info_from_spike_info(spike_info_after_sorting, kilosort_folder,
                                                                    sampling_freq)
# OR load it if already made
template_info = np.load(join(kilosort_folder, 'template_info.df'), allow_pickle=True)

# b) Run the following command in the command line to create the new avg_spike_template.npy
# E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis>python create_data_cubes.py
#                                                                                       infos
#                                                                                       D:\\Data\\George\\AK_33.1\\2018_04_30-11_38\\Analysis\\Denoised\\Kilosort
#                                                                                       D:\\Data\\George\\AK_33.1\\2018_04_30-11_38\\Analysis\\Denoised\\Data\\Amplifier_APs_Denoised.bin
#                                                                                       D:\\Data\\George\\AK_33.1\\2018_04_30-11_38\\Analysis\\Denoised\\Kilosort\\spike_info_after_cortex_sorting.df
#                                                                                       D:\\Data\\George\\AK_33.1\\2018_04_30-11_38\\Analysis\\Denoised\\Kilosort\\template_info.df
#                                                                                       1368
#                                                                                       30
# (Use single space between parameters, not Enter like here)
# (Change the folders as appropriate for where the data is)

# c) Recreate the positions of each template given the new template_info and avg_spike_template files
avg_spike_template = np.load(join(kilosort_folder, 'avg_spike_template.npy'))
template_positions = spp.generate_probe_positions_of_templates(kilosort_folder,
                                                               new_templates_array=avg_spike_template)

# d) , have a look at them
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info)

# e) and finally every time the positions are updated the generate_template_info_from_spike_info must be run to update
# the template_info position columns
template_info = preproc_kilo.generate_template_info_from_spike_info(spike_info_after_sorting, kilosort_folder,
                                                                    sampling_freq)

