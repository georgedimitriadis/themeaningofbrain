

from os.path import join
import numpy as np
import pickle

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis.Graphics import ploting_functions as pf
import matplotlib.pyplot as plt

import pandas as pd
from spikesorting_tsne import io_with_cpp as tsne_io

import slider as sl

import cv2

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

tsne_spikes_folder_base = join(analysis_folder, 'Tsnes', 'SpikingVectors')
tsne_video_folder_base = join(analysis_folder, 'Tsnes', 'Video')

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

results_folder = join(analysis_folder, 'Results')

events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)

regressions_folder = join(results_folder, 'Regressions', 'DistanceTraveledBetweenPokes')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_full_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')
video_cropped_file = join(dlc_folder, 'BonsaiCroping', 'Croped_150_video.avi')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

time_to_bin = 0.1
frames_to_bin = time_to_bin * 120

#   Define the folder for the t-snes
tsne_video_folder = join(tsne_video_folder_base, 'CroppedVideo_100ms_Top100PCs')
tsne_spikes_folder = join(tsne_spikes_folder_base, 'All_spikes_100msbin_count_top40PCs_10Kiter')

#   Load t-sne specific stuff
tsne_spikes = tsne_io.load_tsne_result(tsne_spikes_folder)

tsne_pcs_spike_count_image_file = join(analysis_folder, 'Images', 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_spike_count_image = plt.imread(tsne_pcs_spike_count_image_file)
tsne_pcs_spike_count_image_extent = [-16, 21, -17, 14]


tsne_video = tsne_io.load_tsne_result(tsne_video_folder)

dbscan_result_file = join(tsne_video_folder, 'dbscan_result.pcl')
dbscan_result = pickle.load(open(dbscan_result_file, 'rb'))

tsne_pcs_video_image_file = join(tsne_video_folder, 'Tsne_of_100_top_PCs_of_cropped_video.png')
tsne_pcs_video_image = plt.imread(tsne_pcs_video_image_file)
tsne_pcs_video_image_extent = [-77, 76, -80, 80]

reversed_pcs_video = np.load(join(tsne_video_folder, 'rev_pcs_video.npy'))

video_cropped_resolution = (80, 80)
capture_cropped = cv2.VideoCapture(video_cropped_file)
num_of_frames = int(capture_cropped.get(cv2.CAP_PROP_FRAME_COUNT))

video_full_resolution = (100, 80)
capture_full = cv2.VideoCapture(video_full_file)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="VIEW DYNAMICALLY BOTH VIDEO AND SPIKE TSNES">

video_tsne_labels = dbscan_result.labels_
label_colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(set(video_tsne_labels))))


def show_video_and_pc_reconstructed_frames(i, ax_video, ax_pcs):
    ax_video.clear()
    ax_pcs.clear()
    capture_cropped.set(cv2.CAP_PROP_POS_FRAMES, int(i*frames_to_bin))
    _, video_frame = capture_cropped.read()
    ax_video.imshow(video_frame)
    ax_pcs.imshow(reversed_pcs_video[i, :, :])


f_video = plt.figure(4)
ax_video = f_video.add_subplot(111)
f_pcs = plt.figure(5)
ax_pcs = f_pcs.add_subplot(111)

arg = [ax_video, ax_pcs]
pf.show_clustered_tsne(dbscan_result, tsne_video,
                       func_to_exec_on_pick=show_video_and_pc_reconstructed_frames, args_of_func_on_pick=arg)


def show_video_label_on_spikes_tsne(label, ax2):
    ax2.clear()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    indices = np.squeeze(np.argwhere(video_tsne_labels == label))
    ax2.scatter(tsne_spikes[indices, 0], tsne_spikes[indices, 1], c=[label_colors[label]])
    return None


label = 0
out = None
fig_scat = plt.figure(0)
ax1 = fig_scat.add_subplot(111)
ax1.imshow(tsne_pcs_spike_count_image, extent=tsne_pcs_spike_count_image_extent, aspect='auto')
ax2 = ax1.twinx()
args = [ax2]
sl.connect_repl_var(globals(), 'label', 'out', 'show_video_label_on_spikes_tsne', 'args',
                    slider_limits=[0, video_tsne_labels.max()-1])


# </editor-fold>
# -------------------------------------------------
