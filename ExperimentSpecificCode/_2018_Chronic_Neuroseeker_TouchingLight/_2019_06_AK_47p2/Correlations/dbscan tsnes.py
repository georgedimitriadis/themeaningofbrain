
from os.path import join
import numpy as np
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis import tsne_analysis_functions as tsne_funcs
from BrainDataAnalysis import binning
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
from spikesorting_tsne import tsne, io_with_cpp as tsne_io

import sequence_viewer as sv
import transform as tr
import slider as sl

from sklearn import cluster

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

tsne_folder_base = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder],
                   'Analysis', 'Tsnes', 'SpikingVectors')

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)


spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_video_frame = np.load(spike_rates_per_video_frame_filename)


dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')
body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_10Kiter')

tsne_result = tsne_io.load_tsne_result(tsne_folder).transpose()


# </editor-fold>
# -------------------------------------------------

db, n_clusters_, labels, core_samples_mask, score = tsne_funcs.fit_dbscan(tsne_result, eps=0.032, min_samples=150,
                                                                          normalize=False, show=True)