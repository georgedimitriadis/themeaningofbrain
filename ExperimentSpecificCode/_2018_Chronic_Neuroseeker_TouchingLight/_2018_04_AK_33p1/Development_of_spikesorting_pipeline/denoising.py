"""
This module has developed a technique to amplify the regions in the data that correspond to spikes. The final code that
does the denoising (resulting from the development in this script) can be found in
BrainDataAnalysis.Spike_Sorting.dense_probe_denoising.py.

An example on how to call this function is in this script under the comment header 'Test denoising from its own module'
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join

from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, io_with_cpp as tsne_io, tsne as tsne,\
                              positions_on_probe as spp, visualization as viz
from spikesorting_tsne_guis import clean_kilosort_templates as clean

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs

from scipy.signal import hilbert, butter, sosfiltfilt, savgol_filter

import sequence_viewer as seq_v
import transform as tr
import one_shot_viewer as one_s_v

from copy import copy
import dask


# FOLDERS NAMES --------------------------------------------------
date = 8
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date], 'Data')

binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne')
barnes_hut_exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'


raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

sampling_freq = 20000

denoised_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Denoised')

denoised_data_filename = join(denoised_folder, 'Data', 'Amplifier_APs_Denoised.bin')

# -----------------------------------------------------------------


data_cor = raw_data[1050:, :]
data_hyp = raw_data[850:1050, :]
data_th = raw_data[370:850, :]
data_sth = raw_data[:370, :]

buffer = 500
pointer = 20000


def space_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (100*i) for i in np.arange(dat.shape[0])]))
    return result


seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_cor', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_hyp', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_th', transform_name='space_data')

seq_v.graph_range(globals(), 'pointer', 'buffer', 'raw_data', transform_name='space_data')


sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()

video_frame = 0
video_file = join(data_folder, 'Video.avi')
seq_v.image_sequence(globals(), 'video_frame', 'video_file')

# Create some arrays and constants relating to the events
camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
points_per_pulse = np.mean(np.diff(camera_pulses))

camera_frames_in_video = csv_funcs.get_true_frame_array(data_folder)
time_point_of_first_video_frame = camera_pulses[camera_frames_in_video][0]


def time_point_to_frame(x):
    return sync_funcs.time_point_to_frame(time_point_of_first_video_frame, camera_frames_in_video,
                                          points_per_pulse, x)


tr.connect_repl_var(globals(), 'pointer', 'time_point_to_frame', 'video_frame')


# -----------------------------------------------------------------------------
# DETECTING SUBTHRESHOLD SPIKES WITH INSTANTANEOUS PHASE ----------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


example_point = 25978524
step = 500
data = raw_data[:, example_point:example_point+step]
d_data = np.concatenate((np.diff(data, axis=1), np.zeros((data.shape[0], 1))), axis=1)
d_data_smooth = savgol_filter(d_data, 11, 3)

channels = np.arange(610, 660)
channels = np.arange(len(data))

filtered_bp = butter_bandpass_filter(data, 1500, 2500, 20000, 5)
z_f = hilbert(filtered_bp, axis=1)
f_phases = np.cos(np.unwrap(np.angle(z_f)))

#filtered_d_data_smooth = butter_bandpass_filter(d_data_smooth, 1500, 2500, 20000, 5)
d_z = hilbert(d_data_smooth)
d_phases = np.cos(np.unwrap(np.angle(d_z)))

ht = []
phases = []
amplitudes = []
f_phases = []
f_amplitudes = []
for c in range(len(data)):
    z = hilbert(data[c])
    z_f = hilbert(filtered_bp[c])
    ht.append(z)
    phases.append(np.cos(np.unwrap(np.angle(z))))
    amplitudes.append(np.abs(z))
    f_phases.append(np.cos(np.unwrap(np.angle(z_f))))
    f_amplitudes.append(np.abs(z_f))

phases = np.array(phases)
amplitudes = np.array(amplitudes)
f_phases = np.array(f_phases)
f_amplitudes = np.array(f_amplitudes)

sub_channels = range(640, 650)
sub_channels = channels


m_phases = np.abs(np.sum(phases[sub_channels, :], axis=0))
m_phases_f = np.abs(np.sum(f_phases[sub_channels, :], axis=0))
m_phases_f_smoothed = savgol_filter(m_phases_f, 51, 3)
m_phases_d = np.abs(np.sum(d_phases[sub_channels, :], axis=0))
m_phases_d_smoothed = savgol_filter(m_phases_d, 51, 3)

channel_group = 20
phases_m_g = []
phases_m_f_g = []
for c in range(len(data)):
    if c < len(data) - channel_group:
        chans = range(c, c+channel_group)
        phases_m_g.append(np.abs(np.sum(phases[chans, :], axis=0)))
        phases_m_f_g.append(savgol_filter(np.abs(np.sum(f_phases[chans, :], axis=0)), 51, 3))
    else:
        phases_m_g.append(np.zeros(data.shape[1]))
        phases_m_f_g.append(np.zeros(data.shape[1]))


phases_m_g = np.array(phases_m_g)
phases_m_f_g = np.array(phases_m_f_g)

fig = plt.figure(2)
fig.set_size_inches(10, 19)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('viridis')
ax.imshow(phases_m_f_g, origin='lower', aspect='auto')

one_s_v.graph(globals(), 'phases_m_f_g')

one_s_v.image(globals(), 'phases_m_f_g', image_levels=[-2, 15], flip = 'ud')

fig = plt.figure()
ax1 = fig.add_subplot(611)
ax1.plot(d_data_smooth[sub_channels, :].T)
ax2 = fig.add_subplot(612)
ax2.plot(d_phases[sub_channels, :].T)
ax3 = fig.add_subplot(613)
ax3.plot(m_phases_f_smoothed.T)
ax4 = fig.add_subplot(614)
ax4.plot(f_phases[sub_channels, :].T)
ax5 = fig.add_subplot(615)
ax5.plot(m_phases_d.T)
ax6 = fig.add_subplot(616)
ax6.plot(m_phases_d_smoothed.T)


# -------------------------------------------------------------------------
# SEE EVERYTHING ----------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def space_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (100*i) for i in np.arange(dat.shape[0])]))
    return result


def create_spike_detection_mask(data, low_cut=1500, high_cut=2500, fs=20000, order=5, channel_group=50, thrsehold=2):

    filtered_bp = butter_bandpass_filter(data, low_cut, high_cut, fs, order)
    z_f = hilbert(filtered_bp, axis=1)
    phases = np.cos(np.unwrap(np.angle(z_f)))

    d_data = np.concatenate((np.diff(data, axis=1), np.zeros((data.shape[0], 1))), axis=1)
    d_data_smooth = savgol_filter(d_data, 11, 3)

    phases = d_data_smooth * phases
    phases_m_f_g = []

    for c in range(len(data)):
        if c < len(data) - channel_group:
            chans = range(c, c + channel_group)
            phases_m_f_g.append(savgol_filter(np.abs(np.sum(phases[chans, :], axis=0)), 31, 3))
        else:
            phases_m_f_g.append(np.ones(data.shape[1]))

    phases_m_f_g = np.array(phases_m_f_g)

    return phases_m_f_g


def create_masked_data(data, phases):
    return data * phases * 0.01


def time_point_to_phases(time_point):
    data = raw_data[:, time_point:time_point+buffer]
    return create_spike_detection_mask(data)


def time_point_to_masked_data(time_point):
    global phases_m_f_g
    data = raw_data[:, time_point:time_point + buffer]
    return create_masked_data(data, phases_m_f_g)


cor_channels = np.arange(1050, 1368)
data_cor = raw_data[1050:, :]
hyp_channels = np.arange(850, 1050)
data_hyp = raw_data[850:1050, :]
th_channels = np.arange(370, 850)
data_th = raw_data[370:850, :]
sth_channels = np.arange(370)
data_sth = raw_data[:370, :]

buffer = 500
pointer = 39537363


seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_cor', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_hyp', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_th', transform_name='space_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_sth', transform_name='space_data')

phases_m_f_g = []
tr.connect_repl_var(globals(), 'pointer', 'time_point_to_phases', 'phases_m_f_g')

masked_data = []
tr.connect_repl_var(globals(), 'pointer', 'time_point_to_masked_data', 'masked_data')

phases_m_f_g_cor = []
phases_m_f_g_hyp = []
phases_m_f_g_th = []
phases_m_f_g_sth = []


def cut_phases_into_cor_channels(x):
    return np.array(phases_m_f_g[cor_channels, :])


def cut_phases_into_hyp_channels(x):
    return np.array(phases_m_f_g[hyp_channels, :])


def cut_phases_into_th_channels(x):
    return np.array(phases_m_f_g[th_channels, :])


def cut_phases_into_sth_channels(x):
    return np.array(phases_m_f_g[sth_channels, :])


tr.connect_repl_var(globals(), 'pointer', 'cut_phases_into_cor_channels', 'phases_m_f_g_cor')
tr.connect_repl_var(globals(), 'pointer', 'cut_phases_into_hyp_channels', 'phases_m_f_g_hyp')
tr.connect_repl_var(globals(), 'pointer', 'cut_phases_into_th_channels', 'phases_m_f_g_th')
tr.connect_repl_var(globals(), 'pointer', 'cut_phases_into_sth_channels', 'phases_m_f_g_sth')


levels = [-10, 250]

one_s_v.image(globals(), 'phases_m_f_g_cor', image_levels=levels, flip = 'ud')
one_s_v.image(globals(), 'phases_m_f_g_hyp', image_levels=levels, flip = 'ud')
one_s_v.image(globals(), 'phases_m_f_g_th', image_levels=levels, flip = 'ud')
one_s_v.image(globals(), 'phases_m_f_g_sth', image_levels=levels, flip = 'ud')

one_s_v.image(globals(), 'phases_m_f_g', image_levels=levels, flip = 'ud')


masked_data_cor = []
masked_data_hyp = []
masked_data_th = []
masked_data_sth = []


def cut_masked_data_into_cor_channels(x):
    return space_data(np.array(masked_data[cor_channels, :]))


def cut_masked_data_into_hyp_channels(x):
    return space_data(np.array(masked_data[hyp_channels, :]))


def cut_masked_data_into_th_channels(x):
    return space_data(np.array(masked_data[th_channels, :]))


def cut_masked_data_into_sth_channels(x):
    return space_data(np.array(masked_data[sth_channels, :]))


tr.connect_repl_var(globals(), 'pointer', 'cut_masked_data_into_cor_channels', 'masked_data_cor')
tr.connect_repl_var(globals(), 'pointer', 'cut_masked_data_into_hyp_channels', 'masked_data_hyp')
tr.connect_repl_var(globals(), 'pointer', 'cut_masked_data_into_th_channels', 'masked_data_th')
tr.connect_repl_var(globals(), 'pointer', 'cut_masked_data_into_sth_channels', 'masked_data_sth')

one_s_v.graph(globals(), 'masked_data_cor')
one_s_v.graph(globals(), 'masked_data_hyp')
one_s_v.graph(globals(), 'masked_data_th')
one_s_v.graph(globals(), 'masked_data_sth')


'''
Good place for cortex just before
pointer = 39537363
'''

# -------------------------------------------------------------------------
# CREATE FULL BINARY-------------------------------------------------------

number_of_points = sampling_freq * 10

# Run the following only when you create a new file
# denoised_data = np.memmap(denoised_data_filename, const.BINARY_FILE_ENCODING, mode='w+', shape=raw_data.shape, order='F')

# Create full binary using DASK


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def create_spike_detection_mask(data, low_cut=1500, high_cut=2500, fs=20000, order=5, channel_group=50, thrsehold=2):

    filtered_bp = butter_bandpass_filter(data, low_cut, high_cut, fs, order)
    z_f = hilbert(filtered_bp, axis=1)
    phases = np.cos(np.unwrap(np.angle(z_f)))

    d_data = np.concatenate((np.diff(data, axis=1), np.zeros((data.shape[0], 1))), axis=1)
    d_data_smooth = savgol_filter(d_data, 11, 3)

    phases = d_data_smooth * phases
    phases_m_f_g = []

    for c in range(len(data)):
        if c < len(data) - channel_group:
            chans = range(c, c + channel_group)
            phases_m_f_g.append(savgol_filter(np.abs(np.sum(phases[chans, :], axis=0)), 31, 3))
        else:
            phases_m_f_g.append(np.ones(data.shape[1]))

    phases_m_f_g = np.array(phases_m_f_g)
    #std = phases_m_f_g.std()

    #phases_m_f_g[phases_m_f_g < std * thrsehold] = 1

    return phases_m_f_g


def make_start(i):
    return int(i * number_of_points)


def make_end(i):
    return int(number_of_points * i + number_of_points)


def cut_data(start, end):
    print('Cut {}'.format(str(start/number_of_points)))
    raw_data = ns_funcs.load_binary_amplifier_data(binary_data_filename,
                                                   number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)
    data = copy(raw_data[:, int(start):int(end)])
    del raw_data
    return data


def mult_data(data, mask):
    return mask * data


def assign(data, start, end):
    denoised_data = np.memmap(denoised_data_filename, const.BINARY_FILE_ENCODING, mode='r+',
                                        shape=raw_data.shape, order='F')
    denoised_data[:, start:end] = data
    del denoised_data

    print('Done {} of {}'.format(str(start/number_of_points), str(int(raw_data.shape[1]/number_of_points)+1)))

    return end


number_of_points = sampling_freq * 1


def process(i):
    start = int(i * number_of_points)
    end = int(number_of_points * i + number_of_points)

    data = cut_data(start, end)
    mask = create_spike_detection_mask(data)
    new_data = mult_data(data, mask)
    new_data = new_data.astype(np.int16)
    assign(new_data, start, end)


pane_index = np.arange(int(raw_data.shape[1]/number_of_points)+1)

v = [dask.delayed(process)(i) for i in pane_index]

dask.compute(*v)

remaining_points = raw_data.shape[1] - pane_index[-1] * number_of_points


# -------------------------------------------------------------------------
# HAVE A LOOK AT THE DENOISED DATA ----------------------------------------

raw_data_denoised = ns_funcs.load_binary_amplifier_data(denoised_data_filename,
                                               number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)


data_denoised_cor = raw_data_denoised[1050:, :]
data_denoised_hyp = raw_data_denoised[850:1050, :]
data_denoised_th = raw_data_denoised[370:850, :]
data_denoised_sth = raw_data_denoised[:370, :]


def space_denoised_data(dat):
    dat = dat.astype(np.float32)
    result = np.array(([dat[i, :] + (5000*i) for i in np.arange(dat.shape[0])]))
    return result


buffer = 500
pointer = 39537363


seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_cor', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_hyp', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_th', transform_name='space_denoised_data')
seq_v.graph_range(globals(), 'pointer', 'buffer', 'data_denoised_sth', transform_name='space_denoised_data')


# -------------------------------------------------------------------------
# CLEAN DENOISED SPIKESORT (RIGHT AFTER KILOSORT) -------------------------

# To create averages of templates use cmd (because the create_data_cubes doesn't work when called from a REPL):
# E:\Software\Develop\Source\Repos\spikesorting_tsne_guis\spikesorting_tsne_guis>python create_data_cubes.py original D:\Data\George\AK_33.1\2018_04_30-11_38\Analysis\Denoised\Kilosort D:\Data\George\AK_33.1\2018_04_30-11_38\Data\Amplifier_APs.bin 1368 50

# To clean:
kilosort_folder_denoised = join(denoised_folder, 'Kilosort')

clean.cleanup_kilosorted_data(kilosort_folder_denoised,
                              number_of_channels_in_binary_file=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE,
                              binary_data_filename=binary_data_filename,
                              prb_file=const.prb_file,
                              type_of_binary=const.BINARY_FILE_ENCODING,
                              order_of_binary='F',
                              sampling_frequency=20000,
                              num_of_shanks_for_vis=5)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# CREATE TEMPLATE INFO ----------------------------------------------------
spp.generate_probe_positions_of_templates(kilosort_folder_denoised)
spp.view_grouped_templates_positions(kilosort_folder_denoised, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT)

#template_info = preproc_kilo.generate_template_info_after_cleaning(kilosort_folder_denoised, sampling_freq)
template_info = np.load(join(kilosort_folder_denoised, 'template_info.df'))

templates_in_cortex = template_info[template_info['position Y'] * const.POSITION_MULT > const.BRAIN_REGIONS['CA1']]
templates_in_cortex.index = np.arange(len(templates_in_cortex))


templates_in_cortex['number of spikes'].sum()

spikes_in_cortex = np.array([])
for t in templates_in_cortex['spikes in template']:
    spikes_in_cortex = np.concatenate((spikes_in_cortex, t))

# -------------------------------------------------------------------------
# Tsne all spikes in cortex together
# Use the distance to templates as features
tsne_cortex_folder = join(denoised_folder, 'Tsne', 'Cortex')

np.save(join(tsne_cortex_folder, 'indices_of_spikes_used.npy'), spikes_in_cortex)
template_features = preproc_kilo.calculate_template_features_matrix_for_tsne(kilosort_folder_denoised,
                                                                             tsne_cortex_folder,
                                                                             spikes_used_with_original_indexing=spikes_in_cortex)
num_dims = 2
perplexity = 100
theta = 0.3
iterations = 4000
random_seed = 1
verbose = 2
exe_dir = r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'


tsne_results = tsne.t_sne(template_features, files_dir=tsne_cortex_folder, exe_dir=exe_dir, num_dims=num_dims,
                          perplexity=perplexity, theta=theta, iterations=iterations, random_seed=random_seed,
                          verbose=verbose)

tsne_results = tsne.t_sne_from_existing_distances(files_dir=tsne_cortex_folder, data_has_exageration=True, num_dims=num_dims,
                                                  theta=theta, iterations=iterations, random_seed=random_seed,
                                                  verbose=verbose, exe_dir=exe_dir)

spike_info = preproc_kilo.generate_spike_info_from_full_tsne(kilosort_folder=kilosort_folder_denoised, tsne_folder=tsne_cortex_folder)


# OR Load previously run t-sne
tsne_results = tsne_io.load_tsne_result(files_dir=tsne_cortex_folder)

# and previously generated spike_info
spike_info = pd.read_pickle(join(tsne_cortex_folder, 'spike_info.df'))


# Have a look
viz.plot_tsne_of_spikes(spike_info=spike_info, legent_on=False)

# Update the original spike info (created after just cleaning) with the new spike_info information from manually sorting
# on the t-sne
spike_info_after_cleaning = preproc_kilo.generate_spike_info_after_cleaning(kilosort_folder_denoised)
spike_info_cortex_sorted = spike_info
tsne_filename = join(tsne_cortex_folder, 'result.dat')
spike_info_after_cortex_sorting_file = join(kilosort_folder, 'spike_info_after_cortex_sorting.df')
spike_info_after_sorting = preproc_kilo.add_sorting_info_to_spike_info(spike_info_after_cleaning,
                                                                       spike_info_cortex_sorted,
                                                                       tsne_filename=tsne_filename,
                                                                       save_to_file=spike_info_after_cortex_sorting_file)



# ------------------------------------------------------------------------
# Test denoising from its own module

from os.path import join
import BrainDataAnalysis.Spike_Sorting.dense_probe_desnoising as denoise
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const


# FOLDERS NAMES --------------------------------------------------
date = 8
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
denoised_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Denoised')
denoised_data_filename = join(denoised_folder, 'Data', 'Amplifier_APs_Denoised.bin')

sampling_freq = 20000

denoise.denoise_data(binary_data_filename, const.BINARY_FILE_ENCODING, denoised_data_filename,
                     const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE, sampling_freq, compute_window_in_secs=1,
                     use_dask_for_parallel=True)