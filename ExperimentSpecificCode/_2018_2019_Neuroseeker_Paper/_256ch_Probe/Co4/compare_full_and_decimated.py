

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy as sp
import scipy.ndimage

import sequence_viewer as sv
import transform as tr
import one_shot_viewer as osv
import slider as sl

from sklearn.decomposition import PCA

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.Co4.Decimated_to_Smaller_Density\
    import constants as const_deci

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.Co4.All_Channels\
    import constants as const_all

from spikesorting_tsne import tsne, io_with_cpp as tsne_io
from spikesorting_tsne_guis import spike_heatmap as sh

from Layouts.Probes import probes_imec_256channels as probe_256

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "FOLDERS NAMES"

barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'

# Decimated
analysis_folder_desi = join(const_deci.base_save_folder, const_deci.cell_folder,
                       'Analysis')
binary_data_desi_filename = join(analysis_folder_desi, const_deci.decimation_type_folder, r'decimated_data_256channels.bin')
kilosort_desi_folder = join(analysis_folder_desi, const_deci.decimation_type_folder, 'Kilosort')

template_info_desi = np.load(join(kilosort_desi_folder, 'template_info.df'), allow_pickle=True)
spikes_info_desi = np.load(join(kilosort_desi_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
avg_templates_desi = np.load(join(kilosort_desi_folder, 'avg_spike_template.npy'))

channels_to_keep = [163, 249, 36, 12, 216, 201, 63, 50, 168, 145, 125, 106, 170, 186, 89, 104]

template_info_desi_nonMUA = template_info_desi[template_info_desi['type'] != 'MUA']

# All
binary_data_all_filename = join(const_all.base_save_folder, const_all.cell_folder, 'Data', const_all.data_filename)
analysis_all_folder = join(const_all.base_save_folder, const_all
                           .cell_folder,
                       'Analysis')
kilosort_all_folder = join(analysis_all_folder, const_all.decimation_type_folder, 'Kilosort')

template_info_all = np.load(join(kilosort_all_folder, 'template_info.df'), allow_pickle=True)
spikes_info_all = np.load(join(kilosort_all_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
avg_templates_all = np.load(join(kilosort_all_folder, 'avg_spike_template.npy'))

tsne_folder = join(analysis_all_folder, const_all.decimation_type_folder, 'TSNE',
                   'Peak2Peaks_for_cell_discrimination_10PCs_it4000')

template_info_all_nonMUA = template_info_all[template_info_all['type'] != 'MUA']

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = PREPROCESS (CREATE A PROPER DESIMATED DATA SET WITH ONLY 16 CHANNELS, BASELINE, INTERPOLATE MISSING CHANNELS AND SMOOTH

# correct the bad channels by copying the neighbouring good ones to them

i = 0
k = 0
interpolated_avg_templates_desi = np.zeros((avg_templates_desi.shape[0], 256, avg_templates_desi.shape[2]))
while i < interpolated_avg_templates_desi.shape[1]:
    while i in const_all.bad_channels:
        i = i+1
    interpolated_avg_templates_desi[:, i, :] = avg_templates_desi[:, k, :]
    i = i+1
    k = k+1
interpolated_avg_templates_desi[:, const_all.bad_channels, :] = \
    interpolated_avg_templates_desi[:, const_deci.neighbours_to_bad_channels, :]

# baseline and select only one channel from each one of the 16 groups of identical channels
avg_templates_bs_desi = np.zeros((interpolated_avg_templates_desi.shape[0], len(channels_to_keep), interpolated_avg_templates_desi.shape[2]))
for t in np.arange(avg_templates_desi.shape[0]):
    avg_templates_bs_desi[t, :, :] = interpolated_avg_templates_desi[t, channels_to_keep, :] - \
                                     np.expand_dims(np.mean(interpolated_avg_templates_desi[t, channels_to_keep, :10], axis=1), axis=1)


# Baseline the all channels template timeseries
avg_templates_bs_all = np.zeros(avg_templates_all.shape)
for t in np.arange(avg_templates_bs_all.shape[0]):
    avg_templates_bs_all[t, :, :] = avg_templates_all[t, :, :] - \
                                    np.expand_dims(np.mean(avg_templates_all[t, :, :10], axis=1), axis=1)
# Correct the bad channels in the full channels template average by copying the neighbouring good ones to them
i = 0
k = 0
interpolated_avg_templates_all = np.zeros((avg_templates_bs_all.shape[0], 256, avg_templates_bs_all.shape[2]))
while i < interpolated_avg_templates_all.shape[1]:
    while i in const_all.bad_channels:
        i = i + 1
    interpolated_avg_templates_all[:, i, :] = avg_templates_bs_all[:, k, :]
    i = i + 1
    k = k + 1
interpolated_avg_templates_all[:, const_all.bad_channels, :] = \
    interpolated_avg_templates_all[:, const_all.neighbours_to_bad_channels, :]

# Smooth out the result in the full channels tempate average because there seems to be too many channels with different gains
sigma = [1, 1]
smoothed_avg_templates_all = np.zeros(interpolated_avg_templates_all.shape)
for i in np.arange(interpolated_avg_templates_all.shape[0]):
    smoothed_avg_templates_all[i, :, :] = \
        sp.ndimage.filters.gaussian_filter(interpolated_avg_templates_all[i, :, :], sigma, mode='constant')

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = FIND THE MOST SIMILAR TEMPLATES BETWEEN THE FULL CHANNELS AND THE DECIMATED ONES USING SIMILARITY BETWEEN CHANNELS


# Get the most prominent channel from each template
deeper_channels_desi = np.argmin(np.min(avg_templates_bs_desi, axis=2), axis=1)

best_channel_avg_template_bs_desi = np.zeros((avg_templates_bs_desi.shape[0], avg_templates_bs_desi.shape[2]))
for i in np.arange(len(deeper_channels_desi)):
    best_channel_avg_template_bs_desi[i, :] = avg_templates_bs_desi[i, deeper_channels_desi[i], :]


deeper_channels_all = np.argmin(np.min(avg_templates_bs_all, axis=2), axis=1)

best_channel_avg_template_bs_all = np.zeros((avg_templates_bs_all.shape[0], avg_templates_bs_all.shape[2]))
for i in np.arange(len(deeper_channels_all)):
    best_channel_avg_template_bs_all[i, :] = avg_templates_bs_all[i, deeper_channels_all[i], :]

commonality_between_templates = np.zeros((best_channel_avg_template_bs_all.shape[0], best_channel_avg_template_bs_desi.shape[0]))
for a in np.arange(best_channel_avg_template_bs_all.shape[0]):
    for d in np.arange(best_channel_avg_template_bs_desi.shape[0]):
        commonality_between_templates[a, d] = np.dot(best_channel_avg_template_bs_all[a],
                                                     best_channel_avg_template_bs_desi[d])

most_common_template_pair = np.unravel_index(np.argmax(commonality_between_templates), commonality_between_templates.shape)
most_common_template_pair = np.flip(most_common_template_pair)

# 1st axis is the desi and 2nd the all


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = FIND THE MOST SIMILAR TEMPLATES BETWEEN THE FULL CHANNELS AND THE DECIMATED ONES USING SIMILARITY BETWEEN SPIKE SERIES

crosscorrelogram_spike_times = np.zeros((template_info_desi_nonMUA.shape[0], template_info_all_nonMUA.shape[0]))

for d in np.arange(template_info_desi_nonMUA.shape[0]):
    desi_spikes = template_info_desi_nonMUA['spikes in template'].iloc[d]
    desi_times = spikes_info_desi[np.in1d(spikes_info_desi['original_index'], desi_spikes)]['times'].to_numpy()

    for a in np.arange(template_info_all_nonMUA.shape[0]):
        all_spikes = template_info_all_nonMUA['spikes in template'].iloc[a]
        all_times = spikes_info_all[np.in1d(spikes_info_all['original_index'], all_spikes)]['times'].to_numpy()
        crosscorrelogram_spike_times[d, a] = np.correlate(desi_times, all_times, mode='valid')[0]

most_common_template_pair_from_clean = np.unravel_index(np.argmax(crosscorrelogram_spike_times),
                                                        crosscorrelogram_spike_times.shape)

most_common_template_pair = []
most_common_template_pair.append(template_info_desi['template number'].iloc[most_common_template_pair_from_clean[0]])
most_common_template_pair.append(template_info_all['template number'].iloc[most_common_template_pair_from_clean[1]])


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = VISUALISE ALL HEATMAPS AND CHOOSE THE BEST TEMPLATE TO SHOW

template_to_show = 0
output = None
f1 = plt.figure(10)
a1 = f1.add_subplot(111)

def plot_topoplot_all(i):
    t = template_info_all_nonMUA.iloc[i]['template number']
    _ = sh.create_heatmap_on_matplotlib_widget(a1, smoothed_avg_templates_all[t, :, :], const_all.prb_file,
                                                    window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                    flip_lr=True, flip_ud=False)

sl.connect_repl_var(globals(), 'template_to_show', 'output', 'plot_topoplot_all', slider_limits=[0, template_info_all_nonMUA.shape[0]-1])



template_deci = 0
output2 = None
f2 = plt.figure(11)
a2 = f2.add_subplot(111)

prb_file = join(const_deci.probe_layout_folder, 'probe_imec_256channels_decimated_file.txt')
def plot_topoplot_deci(i):
    t = template_info_desi_nonMUA.iloc[i]['template number']
    smoothed_topoplot_all = sh.create_heatmap_on_matplotlib_widget(a2, avg_templates_bs_desi[t, :, :], prb_file,
                                                    window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                    flip_lr=True, flip_ud=False)


sl.connect_repl_var(globals(), 'template_deci', 'output2', 'plot_topoplot_deci', slider_limits=[0, template_info_desi_nonMUA.shape[0]-1])


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = VISUALISE BEST PEAK TO PEAK HEATMAPS

template_all = most_common_template_pair[1]
template_desi = most_common_template_pair[0]

template_all = template_info_all_nonMUA.iloc[44]['template number']


# Make topoplot of desimated probe
topoplot_desi = np.flipud(np.reshape(sh.peaktopeak(avg_templates_bs_desi[template_desi, :, :], window_size=40)[4], (4,4)))


# Make topoplot of interpolated full probe
interpolated_topoplot_all = sh.create_heatmap_image(interpolated_avg_templates_all[template_all, :, :], const_all.prb_file,
                                                    window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                    flip_lr=False, flip_ud=False, gridscale=10, width=10, height=10)[0]

# Make topoplot of interpolated full probe smoothed
smoothed_topoplot_all = sh.create_heatmap_image(smoothed_avg_templates_all[template_all, :, :], const_all.prb_file,
                                                window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                flip_lr=False, flip_ud=False, gridscale=10, width=10, height=10)[0]

# Show
f0 = plt.figure(0)
f0.add_subplot()
_=plt.imshow(topoplot_desi, interpolation='bicubic')


f1 = plt.figure(1)
f1.add_subplot()
_=plt.imshow(interpolated_topoplot_all, interpolation='bicubic')


f2 = plt.figure(2)
f2.add_subplot()
_=plt.imshow(smoothed_topoplot_all, interpolation='bicubic')


# Show with timeseries superimposed
    # For All
data_used = smoothed_avg_templates_all
f3 = plt.figure(3)
a3 = f3.add_subplot(111)
_ = a3.imshow(smoothed_topoplot_all, interpolation='bicubic')
probe_all = sh.get_probe_geometry_from_prb_file(const_all.prb_file)[0]['geometry']

inset_axis = []
min = data_used[template_all, :, :].min()
max = data_used[template_all, :, :].max()
for c in np.arange(data_used.shape[1]):
    w = probe_all[c][1] / 155 + 0.35/15
    h = probe_all[c][0] / 170 + 0.4/17
    inset_axis.append(a3.inset_axes([w, h-0.025*h, 1/15, 1/17]))
    inset_axis[-1].axis('off')
    inset_axis[-1].set_ylim(min, max)
    if c != 31:
        inset_axis[-1].plot(data_used[template_all, c, :], c='k')

# p2p_from_template_all = sh.peaktopeak(data_used[template_all, :, :], 40)[4]
# decimated_from_p2p = np.flipud(np.reshape(p2p_from_template_all[channels_to_keep], (4,4)))

decimated_data_used = np.zeros((16, data_used.shape[2]))
for g in np.arange(len(const_deci.group_channels)):
    decimated_data_used[g, :] = np.mean(data_used[template_all, const_deci.group_channels[g], :], axis=0)
p2p_from_decimated_data_used = sh.peaktopeak(decimated_data_used, 40)[4]
reshaped_from_decimated_p2p = np.flipud(np.reshape(p2p_from_decimated_data_used, (4, 4)))

f4 = plt.figure(4)
a4 = f4.add_subplot(111)
_ = a4.imshow(reshaped_from_decimated_p2p, interpolation='bicubic', vmin=reshaped_from_decimated_p2p.min(),
              vmax=reshaped_from_decimated_p2p.max())
prb_file = join(const_deci.probe_layout_folder, 'probe_imec_256channels_decimated_file.txt')
probe_desi = sh.get_probe_geometry_from_prb_file(prb_file)[0]['geometry']

inset_axis = []
min = data_used[template_all, channels_to_keep, :].min()
max = data_used[template_all, channels_to_keep, :].max()
for c in np.arange(len(channels_to_keep)):
    w = probe_desi[c][1] / 4
    h = probe_desi[c][0] / 4
    inset_axis.append(a4.inset_axes([w, h, 1 / 4, 1 / 4]))
    inset_axis[-1].axis('off')
    inset_axis[-1].set_ylim(min, max)
    inset_axis[-1].plot(decimated_data_used[c, :], c='k')

    # For Desi
f4 = plt.figure(4)
a4 = f4.add_subplot(111)
_ = a4.imshow(topoplot_desi, interpolation='bicubic', vmin=topoplot_desi.min(), vmax=topoplot_desi.max())
prb_file = join(const_deci.probe_layout_folder, 'probe_imec_256channels_decimated_file.txt')
probe_desi = sh.get_probe_geometry_from_prb_file(prb_file)[0]['geometry']

inset_axis = []
min = avg_templates_bs_desi[template_desi, :, :].min()
max = avg_templates_bs_desi[template_desi, :, :].max()
for c in np.arange(avg_templates_bs_desi.shape[1]):
    w = probe_desi[c][1] / 4
    h = probe_desi[c][0] / 4
    inset_axis.append(a4.inset_axes([w, h, 1/4, 1/4]))
    inset_axis[-1].axis('off')
    inset_axis[-1].set_ylim(min, max)
    inset_axis[-1].plot(avg_templates_bs_desi[template_desi, c, :], c='k')


# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = TSNE P2P AVERAGE DATA TO SEE IF CELLS DIFFERENTIATE DIFFERENTLY WITH THE TWO PROBES (ANSWER = NO!)

electrode_positions, _ = probe_256.create_256channels_imec_prb()
data_used = interpolated_avg_templates_all

data_used_flat = data_used.reshape((data_used.shape[0], data_used.shape[1]*data_used.shape[2]))

# Create the peak to peak for the interpolated data
p2p_from_data_used_all_templates = np.zeros((data_used.shape[0], data_used.shape[1]))
for t in np.arange(data_used.shape[0]):
    p2p_from_data_used_all_templates[t, :] = sh.peaktopeak(data_used[t, :, :], 40)[4]

p2p_from_data_used_all_templates_squared = p2p_from_data_used_all_templates[:, electrode_positions]

p2p_from_data_used_non_mua = p2p_from_data_used_all_templates[template_info_all_nonMUA['template number'].to_numpy(np.int), :]

# Create the peak to peak for the decimated data
decimated_data_used = np.zeros((data_used.shape[0], 16, data_used.shape[2]))
for g in np.arange(len(const_deci.group_channels)):
    decimated_data_used[:, g, :] = np.mean(data_used[:, const_deci.group_channels[g], :], axis=1)

decimated_data_used_flat = decimated_data_used.reshape((decimated_data_used.shape[0],
                                                        decimated_data_used.shape[1]*decimated_data_used.shape[2]))

p2p_from_decimated_data_used_all_templates = np.zeros((decimated_data_used.shape[0], decimated_data_used.shape[1]))
for t in np.arange(decimated_data_used.shape[0]):
    p2p_from_decimated_data_used_all_templates[t, :] = sh.peaktopeak(decimated_data_used[t, :, :], 40)[4]

p2p_from_decimated_data_used_all_templates_squared = \
    np.flipud(np.reshape(p2p_from_decimated_data_used_all_templates, (p2p_from_decimated_data_used_all_templates.shape[0], 4, 4)))

p2p_from_decimated_data_used_non_mua = p2p_from_decimated_data_used_all_templates[template_info_all_nonMUA['template number'].to_numpy(np.int), :]

# Run the t-Sne for the decimated peak to peak data

pca_sr_deci = PCA()
pcs_ar_deci = pca_sr_deci.fit_transform(decimated_data_used_flat)

number_of_top_pcs = 10
num_dims = 2
perplexity = 2
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_result_deci = tsne.t_sne(pcs_ar_deci, tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_deci = tsne_io.load_tsne_result(tsne_folder)

plt.figure(1)
plt.scatter(tsne_result_deci[:, 0], tsne_result_deci[:, 1], s=3, c='k')

# Run the t-Sne for the full peak to peak data

pca_sr_full = PCA()
pcs_ar_full = pca_sr_full.fit_transform(data_used_flat)

number_of_top_pcs = 10
num_dims = 2
perplexity = 2
theta = 0.3
eta = 200
exageration = 12
iterations = 4000
random_seed = 1
verbose = 2
tsne_result_full = tsne.t_sne(pcs_ar_full[:, :number_of_top_pcs], tsne_folder, barnes_hut_exe_dir,
                                         num_dims=num_dims,
                                         perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                                         iterations=iterations, random_seed=random_seed, verbose=verbose)

tsne_result_full = tsne_io.load_tsne_result(tsne_folder)

plt.figure(2)
plt.scatter(tsne_result_full[:, 0], tsne_result_full[:, 1], s=3, c='k')

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------
