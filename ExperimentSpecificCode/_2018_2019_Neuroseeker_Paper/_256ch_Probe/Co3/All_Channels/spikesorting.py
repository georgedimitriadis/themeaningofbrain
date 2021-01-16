
import numpy as np
from os.path import join
import pandas as pd

from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from spikesorting_tsne_guis import clean_kilosort_templates as clean
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.Co3.All_Channels import constants as const


from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, io_with_cpp as tsne_io, tsne as tsne, \
    visualization as viz
# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "FOLDERS NAMES"
binary_data_filename = join(const.base_save_folder, const.cell_folder, 'Data', const.data_filename)
analysis_folder = join(const.base_save_folder, const.cell_folder,
                       'Analysis')
kilosort_folder = join(analysis_folder, const.decimation_type_folder, 'Kilosort')

tsne_folder = join(analysis_folder, const.decimation_type_folder, 'TSNE')
tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'


sampling_freq = const.SAMPLING_FREQUENCY

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#  STEP 1: RUN KILOSORT ON THE DATA
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "STEP 2: CLEAN SPIKESORT (RIGHT AFTER KILOSORT)"

# a) Create average of templates:
# To create averages of templates use cmd (because the create_data_cubes doesn't work when called from a REPL):
# Go to where the create_data_cubes.py is (in spikesort_tsne_guis/spikesort_tsen_guis) and run the following python command
# (you can use either the raw or the denoised data to create the average)
# python E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis\create_data_cubes.py
#                                                                                original
#                                                                                "D:\_256channels\Co2\Analysis\Full_Channels\Kilosort"
#                                                                                "D:\_256channels\Co2\Data\amplifier2017-02-08T15_34_04.bin"
#                                                                                256
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
                              num_of_shanks_for_vis=1)

# c) Remove some types
template_marking = np.load(join(kilosort_folder, 'template_marking.npy'))
print('Noise: {}'.format(len(np.argwhere(template_marking == 0))))
print('Single: {}'.format(len(np.argwhere(template_marking == 1))))
print('Contaminated: {}'.format(len(np.argwhere(template_marking == 2))))
print('Putative: {}'.format(len(np.argwhere(template_marking == 3))))
print('Multi: {}'.format(len(np.argwhere(template_marking == 4))))
print('Non Multi: {}'.format(len(np.argwhere(template_marking == 1)) + len(np.argwhere(template_marking == 2))
                             +len(np.argwhere(template_marking == 3))))

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "STEP 2: CREATE TEMPLATE INFO OF ALL THE CLEANED TEMPLATES"

# a) Create the positions of the templates on the probe (and have a look)
_ = spp.generate_probe_positions_of_templates(kilosort_folder)
bad_channel_positions = spp.get_y_spread_regions_of_bad_channel_groups(kilosort_folder, const.bad_channels)


# b) Create the template_info.df dataframe (or load it if you already have it)
# template_info = preproc_kilo.generate_template_info_after_cleaning(kilosort_folder, sampling_freq)
template_info = np.load(join(kilosort_folder, 'template_info.df'), allow_pickle=True)

# c) Make the spike info from the initial, cleaned, kilosort results
# spike_info = preproc_kilo.generate_spike_info_after_cleaning(kilosort_folder)
spike_info = np.load(join(kilosort_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = STEP 3: T-SNE THE TEMPLATES

# a) Pick the good spikes
spikes_of_good_templates = np.array([])
for t in template_info['spikes in template']:
    spikes_of_good_templates = np.concatenate((spikes_of_good_templates, t))


# b) Make the distance of each spike to its template's closest templates as the feature matrix of T-sne
np.save(join(tsne_folder, 'indices_of_spikes_used.npy'), spikes_of_good_templates)  # Very important to save this file!
template_features = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder,
                                                                             tsne_folder,
                                                                             spikes_used_with_original_indexing=spikes_of_good_templates)

# c) Run the T-sne
num_dims = 2
perplexity = 100
theta = 0.3
iterations = 4000
random_seed = 1
verbose = 2

# This is a full T-sne (calculates distances in GPU and the T-sne embedding in CPU)
tsne_results = tsne.t_sne(template_features, files_dir=tsne_folder, exe_dir=tsne_exe_dir, num_dims=num_dims,
                          perplexity=perplexity, theta=theta, iterations=iterations, random_seed=random_seed,
                          verbose=verbose)


# OR load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_folder)

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = STEP 4: HAVE A LOOK AT THE T-SNE DATA

# a) Make a spike_info from the t-sne data (will be used later in the t-sne sorting gui)
spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder,
                                                             tsne_folder=tsne_folder)

# OR load a previously generated spike_info
spike_info = pd.read_pickle(join(tsne_folder, 'spike_info.df'))

# and have a look
viz.plot_tsne_of_spikes(spike_info=spike_info, legent_on=False)

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

