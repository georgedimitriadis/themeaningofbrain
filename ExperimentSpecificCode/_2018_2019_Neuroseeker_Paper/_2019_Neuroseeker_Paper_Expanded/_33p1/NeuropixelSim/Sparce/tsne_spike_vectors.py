

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._2019_Neuroseeker_Paper_Expanded._33p1 \
    import constants_33p1 as const_rat

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions import events_sync_funcs as \
    sync_funcs
from BrainDataAnalysis.Statistics import binning

from spikesorting_tsne import tsne, io_with_cpp as tsne_io

import sequence_viewer as sv
import transform as tr
import slider as sl

from sklearn.decomposition import PCA
from sklearn import preprocessing

import cv2

# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">
date = 1
binary_data_filename = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
analysis_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date],
                       'Analysis')
kilosort_folder = join(analysis_folder, 'NeuropixelSimulations', 'Sparce', 'Kilosort')

tsne_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

data_folder = join(const_rat.base_save_folder, const_rat.rat_folder, const_rat.date_folders[date], 'Data')

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

results_folder = join(analysis_folder, 'NeuropixelSimulations', 'Sparce', 'Results')

tsne_folder_base = join(results_folder, 'Tsnes', 'SpikingVectors')

events_folder = join(data_folder, "events")
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)


spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_video_frame = np.load(spike_rates_per_video_frame_filename)


dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')
body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

time_to_bin = 0.1
frames_to_bin = time_to_bin * 120


# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX, PCA IT AND RUN T-SNE ON THE PCs">

tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_6Kiter')

spike_rates_max_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.max, frames_to_bin,
                                                       frames_to_bin)

spike_rates_binary_0p1 = np.copy(spike_rates_max_0p1)
spike_rates_binary_0p1[spike_rates_max_0p1 > 0] = 1
spike_rates_binary_0p1 = spike_rates_binary_0p1.transpose()

pca_sr_bin_0p1 = PCA()
pcs_ar_bin_0p1 = pca_sr_bin_0p1.fit_transform(spike_rates_binary_0p1)

number_of_top_pcs = 40
num_dims = 2
perplexity = 100
theta = 0.3
eta = 200
exageration = 12
iterations = 6000
random_seed = 1
verbose = 1
tsne_spike_rates_binary_pcs_0p1 = tsne.t_sne(pcs_ar_bin_0p1[:, :number_of_top_pcs], tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result = tsne_io.load_tsne_result(tsne_folder)

plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, c='k')

tsne_pcs_count_image_file = join(tsne_folder, 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_count_image = plt.imread(tsne_pcs_count_image_file)
tsne_pcs_count_image_extent = [-9, 7, -6, 5]
# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SHOW TSNE CORRELATED WITH VIDEO TSNE AND COMPARE TO FULL PROBE EQUIVALENT">
tsne_pcs_count_image_file = join(tsne_folder, 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_count_image = plt.imread(tsne_pcs_count_image_file)
tsne_pcs_count_image_extent = [-9, 7, -6, 5]

ns_tsne_pcs_count_image_file = join(analysis_folder, 'Images', 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
ns_tsne_pcs_count_image = plt.imread(ns_tsne_pcs_count_image_file)
ns_tsne_pcs_count_image_extent = [-16, 22, -18, 13]

ns_tsne_result = tsne_io.load_tsne_result(join(const_rat.base_save_folder, const_rat.rat_folder,
                                               const_rat.date_folders[date],
                   'Analysis', 'Tsnes', 'SpikingVectors', 'All_spikes_100msbin_count_top40PCs_6Kiters'))

video_tsne_labels_file = join(analysis_folder, 'Tsnes', 'Video', 'CroppedVideo_100ms_Top100PCs', 'dbscan_labels.npy')
video_tsne_labels = np.load(video_tsne_labels_file)

jet = cm.jet
colors = jet(np.squeeze(preprocessing.normalize([video_tsne_labels], 'max')))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=30)


def show_video_label_on_spikes_tsne(label, ax2, ax4):
    indices = np.squeeze(np.argwhere(video_tsne_labels == label))

    ax2.clear()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.scatter(tsne_result[indices, 0], tsne_result[indices, 1], c='r', s=10)

    ax4.clear()
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_ylim(ax3.get_ylim())
    ax4.scatter(ns_tsne_result[indices, 0], ns_tsne_result[indices, 1], c='r', s=10)
    return None


label = 0
out = None
fig_scat = plt.figure(0)
ax1 = fig_scat.add_subplot(111)
ax1.imshow(tsne_pcs_count_image, extent=tsne_pcs_count_image_extent, aspect='auto')
ax2 = ax1.twinx()

fig_scat_ns = plt.figure(1)
ax3 = fig_scat_ns.add_subplot(111)
ax3.imshow(ns_tsne_pcs_count_image, extent=ns_tsne_pcs_count_image_extent, aspect='auto')
ax4 = ax3.twinx()

args = [ax2, ax4]
sl.connect_repl_var(globals(), 'label', 'out', 'show_video_label_on_spikes_tsne', 'args',
                    slider_limits=[0, video_tsne_labels.max()-1])

48, 60, 68, 77, 86, 89, 98, 101, 103
56, 63, 76, 79, 83, 84

f = plt.figure(10)
a1 = f.add_subplot(111)
a1.imshow(ns_tsne_pcs_count_image, extent=ns_tsne_pcs_count_image_extent, aspect='auto')
a2 = a1.twinx()

ind = np.empty(1)
for i in [94, 95, 96, 97, 98]:
    ind = np.concatenate((ind, np.squeeze(np.argwhere(video_tsne_labels == i))))
ind = ind[1:].astype(int)
a2.set_xlim(a1.get_xlim())
a2.set_ylim(a1.get_ylim())
a2.scatter(ns_tsne_result[ind, 0], ns_tsne_result[ind, 1], c='r', s=10)
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
        ax2.scatter(tsne_result[start:stop, 0], tsne_result[start:stop, 1],
                    c=colors)
        for s in np.arange(0, stop - start - 1, 1):
            ax2.plot(tsne_result[start + s:start + s + 2, 0],
                     tsne_result[start + s:start + s + 2, 1],
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
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=10)

# </editor-fold>
# -------------------------------------------------


