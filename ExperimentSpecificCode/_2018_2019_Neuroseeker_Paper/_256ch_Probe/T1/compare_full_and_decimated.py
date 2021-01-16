

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


from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.T1.Decimated_to_Neuroseeker_Density\
    import constants as const_deci

from ExperimentSpecificCode._2018_2019_Neuroseeker_Paper._256ch_Probe.T1.All_Channels\
    import constants as const_all

from spikesorting_tsne_guis import spike_heatmap as sh

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = "FOLDERS NAMES"
# All
binary_data_all_filename = join(const_all.base_save_folder, const_all.cell_folder, 'Data', const_all.data_filename)
analysis_all_folder = join(const_all.base_save_folder, const_all
                           .cell_folder,
                       'Analysis')
kilosort_all_folder = join(analysis_all_folder, const_all.decimation_type_folder, 'Kilosort')

template_info_all = np.load(join(kilosort_all_folder, 'template_info.df'), allow_pickle=True)
spikes_info_all = np.load(join(kilosort_all_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
avg_templates_all = np.load(join(kilosort_all_folder, 'avg_spike_template.npy'))


template_info_all_nonMUA = template_info_all[template_info_all['type'] != 'MUA']

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# <editor-fold desc = PREPROCESS (BASELINE, INTERPOLATE MISSING CHANNELS AND SMOOTH)

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
# <editor-fold desc = VISUALISE

template_all = template_info_all_nonMUA.iloc[18]['template number']


# Make topoplot of interpolated full probe
interpolated_topoplot_all = sh.create_heatmap_image(interpolated_avg_templates_all[template_all, :, :], const_all.prb_file,
                                                    window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                    flip_lr=False, flip_ud=False, gridscale=10, width=10, height=10)[0]

# Make topoplot of interpolated full probe smoothed
smoothed_topoplot_all = sh.create_heatmap_image(smoothed_avg_templates_all[template_all, :, :], const_all.prb_file,
                                                window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                                                flip_lr=False, flip_ud=False, gridscale=10, width=10, height=10)[0]

# Show
'''
f1 = plt.figure(1)
f1.add_subplot()
_=plt.imshow(interpolated_topoplot_all, interpolation='bicubic')


f2 = plt.figure(2)
f2.add_subplot()
_=plt.imshow(smoothed_topoplot_all, interpolation='bicubic')
'''

# Show with timeseries superimposed
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
min = decimated_data_used.min()
max = decimated_data_used.max()
for c in np.arange(len(decimated_data_used)):
    w = probe_desi[c][1] / 4
    h = probe_desi[c][0] / 4
    inset_axis.append(a4.inset_axes([w, h, 1 / 4, 1 / 4]))
    inset_axis[-1].axis('off')
    inset_axis[-1].set_ylim(min, max)
    inset_axis[-1].plot(decimated_data_used[c, :], c='k')

# </editor-fold>
# ----------------------------------------------------------------------------------------------------------------------

template= 0
output = None
f3 = plt.figure(3)
a3 = f3.add_subplot(111)
def create_image(templ):
    image = sh.create_heatmap_on_matplotlib_widget(a3, smoothed_avg_templates_all[templ, :, :], const_all.prb_file,
                            window_size=40, bad_channels=None, num_of_shanks=1, rotate_90=False,
                            flip_lr=False, flip_ud=False)

sl.connect_repl_var(globals(), 'template', 'output', 'create_image')




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



