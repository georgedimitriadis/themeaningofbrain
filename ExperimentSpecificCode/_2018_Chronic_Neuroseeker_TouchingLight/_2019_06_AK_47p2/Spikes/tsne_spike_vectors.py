
from os.path import join
import numpy as np
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from BrainDataAnalysis.Statistics import binning
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
from spikesorting_tsne import tsne, io_with_cpp as tsne_io

import sequence_viewer as sv
import transform as tr
import slider as sl

from sklearn.decomposition import PCA
from sklearn import preprocessing

import cv2

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
results_folder = join(analysis_folder, 'Results')

events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)

regressions_folder = join(results_folder, 'Regressions', 'DistanceTraveledBetweenPokes')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_video_frame = np.load(spike_rates_per_video_frame_filename)


dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')
body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')
mi_dtp_shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_960_vs_distance_to_poke.npy'))
mi_spikes_vs_distance_to_poke = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke.npy'))
mi_spikes_vs_dtp_pb = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke_patterned_behaviour.npy'))
mi_dtp_pb_shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_912_vs_distance_to_poke_patterned_behaviour.npy'))
mi_spike_rates_vs_distance_to_travel_to_poke = \
    np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_travel_to_poke.npy'))
max_neuron = np.argmax(mi_spike_rates_vs_distance_to_travel_to_poke)
mi_spike_rates_vs_distance_to_travel_to_poke_shuffled = np.load(join(mutual_information_folder,
                        'shuffled_mut_info_spike_rate_{}_vs_distance_to_travel_to_poke.npy'.
                        format(str(max_neuron))))

results_folder = join(analysis_folder, 'Results')
poke_folder = join(results_folder, 'EventsCorrelations', 'Poke')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

ti_increasing_neurons_on_trial_pokes = np.load(join(poke_folder, 'ti_increasing_neurons_on_trial_pokes.df'),
                                               allow_pickle=True)
ti_decreasing_neurons_on_trial_pokes = np.load(join(poke_folder, 'ti_decreasing_neurons_on_trial_pokes.df'),
                                               allow_pickle=True)
ti_increasing_neurons_on_non_trial_pokes = np.load(join(poke_folder,
                                                        'ti_increasing_neurons_on_non_trial_pokes.df'),
                                                   allow_pickle=True)
ti_decreasing_neurons_on_non_trial_pokes = np.load(join(poke_folder,
                                                        'ti_decreasing_neurons_on_non_trial_pokes.df'),
                                                   allow_pickle=True)
template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

time_to_bin = 0.1
frames_to_bin = time_to_bin * 120

pcs_image_file = join(analysis_folder, 'Images', 'Two_top_PCs_of_0p1_binary_spike_vectors.png')
pcs_image = plt.imread(pcs_image_file)
pcs_image_extent = [-3, 6, -3.5, 6.5]

tsne_pcs_image_file = join(analysis_folder, 'Images', 'Tsne_of_10_top_PCs_of_0p1_binary_spike_vectors.png')
tsne_pcs_image = plt.imread(tsne_pcs_image_file)
tsne_pcs_image_extent = [-30, 35, -37, 30]

tsne_pcs_count_image_file = join(analysis_folder, 'Images', 'Tsne_of_10_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_count_image = plt.imread(tsne_pcs_count_image_file)
tsne_pcs_count_image_extent = [-25, 28, -19, 20]

tsne_pcs_count_image_file = join(analysis_folder, 'Images', 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_count_image = plt.imread(tsne_pcs_count_image_file)
tsne_pcs_count_image_extent = [-16, 21, -17, 14]

tsne_pcs_count_cn_image_file = join(analysis_folder, 'Images', 'Tsne_of_40_PCs_changing_neurons_of_0p1_count_spike_vectors.png')
tsne_pcs_count_cn_image = plt.imread(tsne_pcs_count_cn_image_file)
tsne_pcs_count_cn_image_extent = [-35, 33, -49, 33]

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX AND RUN T-SNE">
tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_bin_raw')

spike_rates_max_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.max, frames_to_bin,
                                                       frames_to_bin)

spike_rates_binary_0p1 = np.copy(spike_rates_max_0p1)
spike_rates_binary_0p1[spike_rates_max_0p1 > 0] = 1
spike_rates_binary_0p1 = spike_rates_binary_0p1.transpose()

num_dims = 2
perplexity = 100
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_spike_rates_binary_0p1 = tsne.t_sne(spike_rates_binary_0p1, tsne_folder, barnes_hut_exe_dir, num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX, PCA IT AND RUN T-SNE ON THE PCs">

tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_10Kiter')

spike_rates_max_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.max, frames_to_bin,
                                                       frames_to_bin)

spike_rates_binary_0p1 = np.copy(spike_rates_max_0p1)
spike_rates_binary_0p1[spike_rates_max_0p1 > 0] = 1
spike_rates_binary_0p1 = spike_rates_binary_0p1.transpose()

pca_sr_bin_0p1 = PCA()
pcs_ar_bin_0p1 = pca_sr_bin_0p1.fit_transform(spike_rates_binary_0p1)

number_of_top_pcs = 10
num_dims = 2
perplexity = 100
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_spike_rates_binary_pcs_0p1 = tsne.t_sne(pcs_ar_bin_0p1[:, :number_of_top_pcs], tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms, PCA THIS AND RUN T-SNE">
tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top500PCs_4Kiter')
time_to_bin = 0.1
frames_to_bin = time_to_bin * 120

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

spike_rates_count_0p1 = spike_rates_count_0p1.transpose() * 0.00833

pca_sr_count_0p1 = PCA()
pcs_ar_count_0p1 = pca_sr_count_0p1.fit_transform(spike_rates_count_0p1).astype(np.int16)


number_of_top_pcs = 500
num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 3
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_0p1[:, :number_of_top_pcs], tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms ONLY FOR INCREASING AND DECREASING NEURONS, PCA THIS AND RUN T-SNE">
tsne_folder = join(tsne_folder_base, 'Changing_neurons_spikes_100msbin_count_top40PCs_6Kiter')

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

changing_neurons_indices = np.unique(np.concatenate((ti_increasing_neurons_on_trial_pokes.index.values,
                                                     ti_decreasing_neurons_on_trial_pokes.index.values,
                                                     ti_increasing_neurons_on_non_trial_pokes.index.values,
                                                     ti_decreasing_neurons_on_non_trial_pokes.index.values)))

spike_rates_count_changing_neurons_0p1 = spike_rates_count_0p1[changing_neurons_indices, :]
spike_rates_count_changing_neurons_0p1 = spike_rates_count_changing_neurons_0p1.transpose() * 0.00833

pca_sr_count_changing_0p1 = PCA()
pcs_ar_count_changing_0p1 = pca_sr_count_changing_0p1.fit_transform(spike_rates_count_changing_neurons_0p1).astype(np.int16)

num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 6000
random_seed = 1
verbose = 2
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_changing_0p1, tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms ONLY FOR NEURONS WITH HIGH MI TO DTP, PCA THIS AND RUN T-SNE">
tsne_folder = join(tsne_folder_base, 'Dtp_neurons_spikes_100msbin_count_top40PCs_6Kiter')

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

confidence_level = 0.99
mean_sh = np.mean(mi_dtp_shuffled)
confi_intervals = mi_dtp_shuffled[int((1. - confidence_level) / 2 * 1000)], \
                  mi_dtp_shuffled[int((1. + confidence_level) / 2 * 1000)]

correlated_neuron_indices = np.squeeze(np.argwhere(mi_spikes_vs_distance_to_poke > mean_sh + confi_intervals[1]))

spike_rates_count_changing_neurons_0p1 = spike_rates_count_0p1[correlated_neuron_indices, :]
spike_rates_count_changing_neurons_0p1 = spike_rates_count_changing_neurons_0p1.transpose() * 0.00833

pca_sr_count_changing_0p1 = PCA()
pcs_ar_count_changing_0p1 = pca_sr_count_changing_0p1.fit_transform(spike_rates_count_changing_neurons_0p1).astype(np.int16)

number_of_top_pcs = 40
num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 6000
random_seed = 1
verbose = 3
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_changing_0p1[:, :number_of_top_pcs], tsne_folder,
                                            barnes_hut_exe_dir,
                                            num_dims=num_dims,
                                            perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                            iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------

# --------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms ONLY FOR NEURONS WITH HIGH MI TO DTP DURING PATTERNED BEHAVIOUR">
tsne_folder = join(tsne_folder_base, 'Dtp_pb_neurons_spikes_100msbin_count_top40PCs_6Kiter')

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

confidence_level = 0.99
mean_sh = np.mean(mi_dtp_pb_shuffled)
confi_intervals = mi_dtp_pb_shuffled[int((1. - confidence_level) / 2 * 1000)], \
                  mi_dtp_pb_shuffled[int((1. + confidence_level) / 2 * 1000)]

correlated_neuron_indices = np.squeeze(np.argwhere(mi_spikes_vs_dtp_pb > mean_sh + confi_intervals[1]))

spike_rates_count_changing_neurons_0p1 = spike_rates_count_0p1[correlated_neuron_indices, :]
spike_rates_count_changing_neurons_0p1 = spike_rates_count_changing_neurons_0p1.transpose() * 0.00833

pca_sr_count_changing_0p1 = PCA()
pcs_ar_count_changing_0p1 = pca_sr_count_changing_0p1.fit_transform(spike_rates_count_changing_neurons_0p1).astype(np.int16)

number_of_top_pcs = 40
num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 6000
random_seed = 1
verbose = 3
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_changing_0p1[:, :number_of_top_pcs], tsne_folder,
                                            barnes_hut_exe_dir,
                                            num_dims=num_dims,
                                            perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                            iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------


# --------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms ONLY FOR NEURONS WITH HIGH MI TO DISTANCE TO TRAVEL TO POKE">
tsne_folder = join(tsne_folder_base, 'Distance_to_travel_to_poke_neurons_spikes_100ms_count_top40PCs_4Kiter')

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

confidence_level = 0.99
mean_sh = np.mean(mi_spike_rates_vs_distance_to_travel_to_poke_shuffled)
confi_intervals = mi_spike_rates_vs_distance_to_travel_to_poke_shuffled[int((1. - confidence_level) / 2 * 1000)], \
                  mi_spike_rates_vs_distance_to_travel_to_poke_shuffled[int((1. + confidence_level) / 2 * 1000)]

correlated_neuron_indices = np.squeeze(np.argwhere(mi_spike_rates_vs_distance_to_travel_to_poke > confi_intervals[1]))

spike_rates_count_distance_to_travel_to_poke_neurons_0p1 = spike_rates_count_0p1[correlated_neuron_indices, :]
spike_rates_count_distance_to_travel_to_poke_neurons_0p1 = spike_rates_count_distance_to_travel_to_poke_neurons_0p1.transpose() * 0.00833

pca_sr_count_distance_to_travel_to_poke_0p1 = PCA()
pcs_ar_count_distance_to_travel_to_poke_0p1 = \
    pca_sr_count_distance_to_travel_to_poke_0p1.fit_transform(spike_rates_count_distance_to_travel_to_poke_neurons_0p1).astype(np.int16)

number_of_top_pcs = 40
num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 3
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_distance_to_travel_to_poke_0p1[:, :number_of_top_pcs], tsne_folder,
                                            barnes_hut_exe_dir,
                                            num_dims=num_dims,
                                            perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                            iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="LOOK AT CORRELATION OF TOP PCs WITH BEHAVIOUR">

spike_rates_max_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.max, frames_to_bin,
                                                       frames_to_bin)

spike_rates_binary_0p1 = np.copy(spike_rates_max_0p1)
spike_rates_binary_0p1[spike_rates_max_0p1 > 0] = 1
spike_rates_binary_0p1 = spike_rates_binary_0p1.transpose()

pca_sr_bin_0p1 = PCA()
pcs_ar_bin_0p1 = pca_sr_bin_0p1.fit_transform(spike_rates_binary_0p1)

# plt.plot(np.arange(0, pcs_ar_bin_0p1.shape[0] * 0.1, 0.1), pcs_ar_bin_0p1[:, 0])

tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_bin_top10PCs')
tsne_spike_rates_binary_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)


spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)
spike_rates_count_0p1 = spike_rates_count_0p1.transpose()
pca_sr_count_0p1 = PCA()
pcs_ar_count_0p1 = pca_sr_count_0p1.fit_transform(spike_rates_count_0p1).astype(np.int16)


tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_10Kiter')
tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)


frame = 0
sv.image_sequence(globals(), 'frame', 'video_file')

def phase_space(frame, ax1, ax2, seconds_to_track):
    ax2.clear()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    stop = int(frame / 12)
    start = stop - int(seconds_to_track * 10)
    if start > 0:
        #start = 0
        colormap = cm.jet
        colors = colormap(np.arange(0, 1, 1/(stop-start)))
        ax2.scatter(tsne_spike_rates_count_pcs_0p1[start:stop, 0], tsne_spike_rates_count_pcs_0p1[start:stop, 1], c=colors)
        for s in np.arange(0, stop-start-1, 1):
            ax2.plot(tsne_spike_rates_count_pcs_0p1[start+s:start+s+2, 0], tsne_spike_rates_count_pcs_0p1[start+s:start+s+2, 1],
                     c=colors[s])
    return None


fig_scat = plt.figure(0)
ax1 = fig_scat.add_subplot(111)
ax1.imshow(tsne_pcs_count_cn_image, extent=tsne_pcs_count_cn_image_extent, aspect='auto')
ax2 = ax1.twinx()
seconds_to_track = 1
out = None
args = [ax1, ax2, seconds_to_track]
tr.connect_repl_var(globals(), 'frame', 'out', 'phase_space', 'args')


fig_plot = plt.figure(1)
fig_plot_zoom = plt.figure(2)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="MAKE VIDEO OF RAT VIDEO WITH THE TSNE">

opencv_rat_video = cv2.VideoCapture(video_file)
total_frames = int(opencv_rat_video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_size_of_video_out = (int(2976), int(1549))
dpi = 100
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
tsne_behaviour_video = cv2.VideoWriter(join(tsne_folder, 'tsne_behaviour_video.avi'), fourcc, 10.0, frame_size_of_video_out)

fig_scat = plt.figure(0, figsize=(frame_size_of_video_out[0]/dpi, (frame_size_of_video_out[1])/dpi), dpi=dpi)
fig_scat.set_tight_layout(True)
ax1 = fig_scat.add_subplot(111)
ax1.set_axis_off()
ax1.imshow(tsne_pcs_count_image, extent=tsne_pcs_count_image_extent, aspect='auto')
ax2 = ax1.twinx()
ax2.set_axis_off()
seconds_to_track = 1

for frame in np.arange(0, total_frames, 12):
    opencv_rat_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, rat_video_frame = opencv_rat_video.read()
    rat_video_frame = rat_video_frame[64:576, :, :]
    rvf_h = rat_video_frame.shape[0]
    rvf_w = rat_video_frame.shape[1]
    rat_video_frame = cv2.resize(rat_video_frame, (int(rvf_w * 1.5), int(rvf_h * 1.5)), interpolation=cv2.INTER_AREA)
    rvf_h = rat_video_frame.shape[0]
    rvf_w = rat_video_frame.shape[1]

    ax2.clear()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax1.set_axis_off()
    ax2.set_axis_off()
    stop = int(frame / 12)
    start = stop - int(seconds_to_track * 10)
    if start > 0:
        colormap = cm.jet
        colors = colormap(np.arange(0, 1, 1 / (stop - start)))
        ax2.scatter(tsne_spike_rates_count_pcs_0p1[start:stop, 0], tsne_spike_rates_count_pcs_0p1[start:stop, 1],
                    c=colors)
        for s in np.arange(0, stop - start - 1, 1):
            ax2.plot(tsne_spike_rates_count_pcs_0p1[start + s:start + s + 2, 0],
                     tsne_spike_rates_count_pcs_0p1[start + s:start + s + 2, 1],
                     c=colors[s])

    fig_scat.canvas.draw()
    w, h = fig_scat.canvas.get_width_height()
    buf = np.fromstring(fig_scat.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    buf_a = np.copy(buf[:, :, 0])
    buf_r = np.copy(buf[:, :, 1])
    buf_g = np.copy(buf[:, :, 2])
    buf_b = np.copy(buf[:, :, 3])
    buf[:, :, 0] = buf_r
    buf[:, :, 1] = buf_g
    buf[:, :, 2] = buf_b
    buf[:, :, 3] = buf_a

    buf[-rvf_h:, -rvf_w:, :3] = rat_video_frame
    cv2.imwrite(join(tsne_folder, 'temp.png'), buf)
    temp = cv2.imread(join(tsne_folder, 'temp.png'))
    tsne_behaviour_video.write(temp)

    if frame % (12*100) == 0:
        print('Done {} frames'.format(str(frame)))

opencv_rat_video.release()
tsne_behaviour_video.release()

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SHOW TSNE CORRELATED WITH SPEED AND DISTANCE">

#   Distance to poke
ball_positions_df = event_dataframes['ev_ball_tracking']
body_positions_normalised = 2 * body_positions / 640 - 1
poke_position = [0.9, 0]
distances_rat_to_poke_all_frames = np.sqrt(
    np.power(body_positions_normalised[:, 0] - poke_position[0], 2) +
    np.power(body_positions_normalised[:, 1] - poke_position[1], 2))

distances_rat_to_poke_all_frames_0p1 = binning.rolling_window_with_step(distances_rat_to_poke_all_frames, np.mean, 12, 12)
distances_rat_to_poke_all_frames_0p1_norm = (distances_rat_to_poke_all_frames_0p1 -
                                             np.min(distances_rat_to_poke_all_frames_0p1)) / \
                                            (np.max(distances_rat_to_poke_all_frames_0p1) -
                                             np.min(distances_rat_to_poke_all_frames_0p1))

jet = cm.jet
colors = jet (distances_rat_to_poke_all_frames_0p1_norm)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=10)

#   Categorical distance
distances_rat_to_poke_all_frames_0p1_halfnorm = distances_rat_to_poke_all_frames_0p1_norm/2
colors = cm.tab10 (distances_rat_to_poke_all_frames_0p1_halfnorm)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=10)

distances_for_log_cat = (distances_rat_to_poke_all_frames_0p1_norm + 0.2) * 5
num_of_categories = 4
categories = np.logspace(0, 1.8, num_of_categories + 1, base=np.e)
dtp_cat = []
for c in np.arange(num_of_categories):
    dtp_cat.append(np.squeeze(np.argwhere(np.logical_and(distances_for_log_cat >= categories[c],
                                                         distances_for_log_cat < categories[c+1]))))

for i in np.arange(num_of_categories):
    colors = np.array([(0, 0, 0, 0.3)] * tsne_result.shape[1])
    colors[dtp_cat[i]] = (1, 0, 0, 1)
    plt.figure(i).add_subplot(111).scatter(tsne_result[0, :], tsne_result[1, :], c=colors, s=50)


#   Speed
speeds_0p1 = binning.rolling_window_with_step(speeds, np.mean, 12, 12)
speeds_0p1_norm = (speeds_0p1 - np.nanmin(speeds_0p1)) / (2 * np.nanmedian(speeds_0p1) - np.nanmin(speeds_0p1))
colors = jet (speeds_0p1_norm)
plt.scatter(tsne_spike_rates_count_pcs_0p1[:, 0], tsne_spike_rates_count_pcs_0p1[:, 1], c=colors, s=10)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SHOW TSNE CORRELATED WITH DISTANCE TRAVELED AWAY FROM POKE">

distance_to_travel_to_poke = np.load(join(regressions_folder, 'distance_to_travel_to_poke.npy'))
frame_regions_away_from_poke = np.load(join(regressions_folder, 'frame_regions_away_from_poke.npy'))
frame_windows_away_from_poke = np.empty(0)
for f in frame_regions_away_from_poke:
    period_frames = np.arange(f[0], f[1]).astype(np.int)
    frame_windows_away_from_poke = np.concatenate((frame_windows_away_from_poke, period_frames))
frame_windows_away_from_poke = frame_windows_away_from_poke.astype(np.int)

distance_to_travel_to_poke_all_frames = np.zeros(spike_rates_per_video_frame.shape[1])
distance_to_travel_to_poke_all_frames[frame_windows_away_from_poke] = distance_to_travel_to_poke
distance_to_travel_to_poke_all_frames_smoothed = \
    binning.rolling_window_with_step(distance_to_travel_to_poke_all_frames, np.mean, frames_to_bin, frames_to_bin)
distance_to_travel_to_poke_all_frames_smoothed_norm = \
    np.squeeze(preprocessing.normalize([distance_to_travel_to_poke_all_frames_smoothed], norm='max'))

jet = cm.jet
colors = jet(distance_to_travel_to_poke_all_frames_smoothed_norm)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=30)

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="SHOW TSNE CORRELATED WITH VIDEO TSNE">

video_tsne_labels_file = join(analysis_folder, 'Tsnes', 'Video', 'CroppedVideo_100ms', 'dbscan_labels.npy')
video_tsne_labels = np.load(video_tsne_labels_file)

jet = cm.jet
colors = jet(np.squeeze(preprocessing.normalize([video_tsne_labels], 'max')))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=30)


def show_video_label_on_spikes_tsne(label, ax2):
    ax2.clear()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    indices = np.squeeze(np.argwhere(video_tsne_labels == label))
    ax2.scatter(tsne_result[indices, 0], tsne_result[indices, 1], c='r')
    return None


label = 0
out = None
fig_scat = plt.figure(0)
ax1 = fig_scat.add_subplot(111)
ax1.imshow(tsne_pcs_count_image, extent=tsne_pcs_count_image_extent, aspect='auto')
ax2 = ax1.twinx()
args = [ax2]
sl.connect_repl_var(globals(), 'label', 'out', 'show_video_label_on_spikes_tsne', 'args',
                    slider_limits=[0, video_tsne_labels.max()-1])

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="SHOW PATTERNED AND NON PATTERNED TRAJECTORIES ON TSNE">


windows_of_patterned_behaviour = np.load(join(patterned_vs_non_patterned_folder, 'windows_of_patterned_behaviour.npy'),
                                         allow_pickle=True)
windows_of_patterned_behaviour = [np.unique(np.array(windows_of_patterned_behaviour[l]/12).astype(np.int))
                                  for l in np.arange(len(windows_of_patterned_behaviour))]

windows_of_non_patterned_behaviour = np.load(join(patterned_vs_non_patterned_folder, 'windows_of_non_patterned_behaviour.npy'),
                                             allow_pickle=True)
windows_of_non_patterned_behaviour = [np.unique(np.array(windows_of_non_patterned_behaviour[l]/12).astype(np.int))
                                      for l in np.arange(len(windows_of_non_patterned_behaviour))]


for i in np.arange(len(windows_of_patterned_behaviour)):
    t = tsne[windows_of_patterned_behaviour[i], :]
    plt.plot(t[:, 0], t[:, 1])


def show_traj(traj):
    t = tsne[windows_of_non_patterned_behaviour[traj], :]
    plt.clf()
    plt.plot(t[:, 0], t[:, 1])
    plt.scatter(tsne[:, 0], tsne[:, 1], c=[(0.2, 0.2, 0.2, 0.5)]*tsne.shape[0], s=3)
    return None

traj = 0
out = None
sl.connect_repl_var(globals(), 'traj', 'out', 'show_traj', slider_limits=[0, 32])

# </editor-fold>
# -------------------------------------------------