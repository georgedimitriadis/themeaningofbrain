
from os.path import join
import numpy as np
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p1 import constants as const
from BrainDataAnalysis.Statistics import binning
import matplotlib.pyplot as plt
from matplotlib import cm

from spikesorting_tsne import tsne, io_with_cpp as tsne_io

from sklearn.decomposition import PCA
from sklearn import preprocessing

import cv2

import sequence_viewer as sv
import transform as tr
import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS">
date_folder = 6

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')

kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                       'Kilosort')

tsne_folder_base = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder],
                   'Analysis', 'Tsnes', 'SpikingVectors')

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_video_frame = np.load(spike_rates_per_video_frame_filename)

dlc_folder = join(analysis_folder, 'Deeplabcut')
# dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_6Kiter')
tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

tsne_pcs_count_image_file = join(analysis_folder, 'Images', 'Tsne_of_40_top_PCs_of_0p1_count_spike_vectors.png')
tsne_pcs_count_image = plt.imread(tsne_pcs_count_image_file)
tsne_pcs_count_image_extent = [-31, 45, -45, 30]

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="CREATE MATRIX OF COUNTS PER 100ms, PCA THIS AND RUN T-SNE">
tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_6Kiter')
time_to_bin = 0.1
frames_to_bin = time_to_bin * 120

spike_rates_count_0p1 = binning.rolling_window_with_step(spike_rates_per_video_frame, np.sum, frames_to_bin,
                                                         frames_to_bin)

spike_rates_count_0p1 = spike_rates_count_0p1.transpose() * 0.00833

pca_sr_count_0p1 = PCA()
pcs_ar_count_0p1 = pca_sr_count_0p1.fit_transform(spike_rates_count_0p1).astype(np.int16)


number_of_top_pcs = 40
num_dims = 2
perplexity = 100
theta = 0.2
eta = 200
exageration = 12
iterations = 6000
random_seed = 1
verbose = 2
tsne_spike_rates_count_pcs_0p1 = tsne.t_sne(pcs_ar_count_0p1[:, :number_of_top_pcs], tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_spike_rates_count_pcs_0p1 = tsne_io.load_tsne_result(tsne_folder)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="LOOK AT CORRELATION OF TOP PCs WITH BEHAVIOUR">

time_to_bin = 0.1
frames_to_bin = time_to_bin * 120

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


tsne_folder = join(tsne_folder_base, 'All_spikes_100msbin_count_top40PCs_6Kiter')
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
ax1.imshow(tsne_pcs_count_image, extent=tsne_pcs_count_image_extent, aspect='auto')
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

frame_size_of_video_out = (int(2976), int(1542))
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
    rat_video_frame = cv2.resize(rat_video_frame, (int(rvf_w * 1.2), int(rvf_h * 1.2)), interpolation=cv2.INTER_AREA)
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

    if frame % 12*100 == 0:
        print('Done {} frames'.format(str(frame)))

opencv_rat_video.release()
tsne_behaviour_video.release()

cv2.destroyAllWindows()


# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="SHOW TSNE CORRELATED WITH VIDEO TSNE">

tsne_result = tsne_io.load_tsne_result(tsne_folder)

video_tsne_labels_file = join(analysis_folder, 'Tsnes', 'Video', 'CroppedVideo_100ms_Top100PCs', 'dbscan_labels.npy')
video_tsne_labels = np.load(video_tsne_labels_file)

jet = cm.jet
colors = jet(np.squeeze(preprocessing.normalize([video_tsne_labels], 'max')))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, s=3)


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


11, 12, 16, 25, 36, 37, 39, 42, 43, 44, 45, 46, 61, 71, 74, 75, 78, 81, 93, 97, 99, 100, 101, 102, 103, 104, 105,
116, 132, 140, 141, 167, 168, 174, 251, 259, 275
# </editor-fold>
# -------------------------------------------------