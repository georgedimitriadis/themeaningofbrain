

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from npeet.lnc import MI
from BrainDataAnalysis.LFP import emd
import common_data_transforms as cdts
from mne.time_frequency import multitaper as mt
from mne.time_frequency import multitaper as mt

from mpl_toolkits import mplot3d

import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from BrainDataAnalysis import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs


import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
dlc_folder = r'D:\Data\George\AK_33.1\2018_04_30-11_38\Analysis\Deeplabcut'
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')
mutual_information_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                                 'Results', 'MutualInformation')

events_folder = join(data_folder, "events")

markers_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1May7shuffle1_150000.h5')

labeled_video_file = join(dlc_project_folder, 'videos', r'Croped_videoDeepCut_resnet50_V1May7shuffle1_150000_labeled.mp4')

crop_window_position_file = join(dlc_folder, 'BonsaiCroping', 'Croped_Video_Coords.csv')

full_video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')


imfs_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder],
                         'Analysis', 'Lfp', 'EMD', 'imfs.bin')

spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

# ------------------------------------------------------
# FIRING RATE DISTRIBUTION
# ------------------------------------------------------
spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))

brain_regions = const.BRAIN_REGIONS
cortex = np.array([brain_regions['Cortex MPA'], brain_regions['CA1']]) / const.POSITION_MULT
hippocampus = np.array([brain_regions['CA1'], brain_regions['Thalamus LPMR']]) / const.POSITION_MULT
thalamus = np.array([brain_regions['Thalamus LPMR'], brain_regions['Zona Incerta']]) / const.POSITION_MULT
sub_thalamic = np.array([brain_regions['Zona Incerta'], 0]) / const.POSITION_MULT

cort_cells = template_info[np.logical_and(template_info['position Y'] < cortex[0], template_info['position Y'] > cortex[1])]
hipp_cells = template_info[np.logical_and(template_info['position Y'] < hippocampus[0], template_info['position Y'] > hippocampus[1])]
thal_cells = template_info[np.logical_and(template_info['position Y'] < thalamus[0], template_info['position Y'] > thalamus[1])]
sub_th_cells = template_info[np.logical_and(template_info['position Y'] < sub_thalamic[0], template_info['position Y'] > sub_thalamic[1])]


plt.hist(sub_th_cells['firing rate'], bins=np.logspace(np.log10(0.001), np.log10(100), 50))
plt.gca().set_xscale("log")


# ------------------------------------------------------
# EMDS
# ------------------------------------------------------

# Load the generated EMD data and have a look
num_of_imfs = const.NUMBER_OF_IMFS
num_of_channels = const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE


imfs = emd.load_memmaped_imfs(imfs_filename, dtype=np.int16, num_of_imfs=num_of_imfs, num_of_channels=num_of_channels)

imf = 0
time = 1000000
buffer = 5000
factor = 2
args = [factor, time, buffer, imfs]


def get_specific_imf_spaced(imf, factor, data):
    imf_data = data[:, imf, :]
    return cdts.space_data_factor(imf_data, factor)


def get_time_window(time, buffer, data):
    return data[:, :, time:time+buffer]


def get_windowed_imf(imf, factor, time, buffer, data):
    return get_specific_imf_spaced(imf, factor, get_time_window(time, buffer, data))


def update_args(t):
    return [factor, t, buffer, imfs]


sl.connect_repl_var(globals(), 'time', 'update_args', 'args', slider_limits=[1000000, 1200000])
window = None
sl.connect_repl_var(globals(), 'imf', 'get_windowed_imf', 'window', 'args', slider_limits=[0, 12])
osv.graph(globals(), 'window')


# Emd spectrum
psd_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder],
                         'Analysis', 'Lfp', 'EMD', 'psd_of_imf_example.npy')
'''
imf_example = imfs[5, :, 1000000:1150000] # 10 minutes
psd_imf, fs = mt.psd_array_multitaper(imf_example, sfreq=const.SAMPLING_FREQUENCY/factor, fmin=1, fmax=3000, bandwidth=6, verbose=0)
np.save(psd_filename, (psd_imf, fs))
'''
psd_imf, fs = np.load(psd_filename, allow_pickle=True)

data = psd_imf
fig = plt.figure(1)
N = int(len(data))
axs=[]
for n in np.arange(N):
    axs.append(fig.add_subplot(N, 1, n+1))
    axs[n].plot(fs, data[n])


# ------------------------------------------------------
# BEHAVIOUR AND SPIKE RATES
# ------------------------------------------------------

updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements(body_positions, maximum_pixels=20)

conversion_const = const.PIXEL_PER_FRAME_TO_METERS_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                         180 * (1/np.pi) * np.arctan(body_velocities[:, 1] / body_velocities[:, 0])]).transpose()



# Look at the whole video with body trajectory superimposed
traj = np.zeros((640, 640, 4))
frame = 3
output = None
def update_trajectory_for_video(frame):
    traj[:, :, :] = 0
    bp = body_positions.astype(int)
    bp[:, 1] = 640 - bp[:, 1]
    traj[bp[:frame, 1], bp[:frame, 0], :] = 255
    return output


tr.connect_repl_var(globals(), 'frame', 'update_trajectory_for_video', 'output')
sv.image_sequence(globals(), 'frame', 'full_video_file', 'traj')
# ------------------------------------------------------

space_occupancy = binning.bin_2d_array(body_positions, bins=[10, 10])

max_pix_per_frame = 20  # Rats run up to 13Km/h = 20 pixels/frame
bins = [np.logspace(np.log10(0.0001), np.log10(max_pix_per_frame * conversion_const), 30, base=10), 20]
velocity_categories = binning.bin_2d_array(body_velocities_polar, bins=bins)
speed_categories = pd.cut(body_velocities_polar[:, 0], bins=bins[0])
movement_categories = pd.cut(body_velocities_polar[:, 0], bins=np.array([0, 0.06, 0.28, 5.7, 70]) * conversion_const,
                             labels=['still', 'jitter', 'walk', 'run'])  # bins in pixels/frame * conversion factor
orientation_categories = pd.cut(body_velocities_polar[:, 1], bins=20)


event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
file_to_save_to = join(spikes_folder, 'firing_rate_with_video_frame_window.npy')
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

spike_rates = np.load(file_to_save_to)
num_of_frames_to_average = 0.25/(1/120)


spike_rates_0p25 = np.load(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'))

speeds_0p25 = binning.rolling_window_with_step(body_velocities_polar[:, 0], np.mean,
                                               num_of_frames_to_average, num_of_frames_to_average)


mutual_infos_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'))
shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.999
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]
plt.hist(mutual_infos_spikes_vs_speed, bins=np.logspace(np.log10(0.0001), np.log10(30), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(30), 50), color=(1, 0, 0, 0.4))
plt.gca().set_xscale("log")
plt.vlines(mean_sh+confi_intervals[1], 0, 60)



speed_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > 0.2))
speed_corr_neurons = template_info.loc[speed_corr_neurons_index]

plt.plot(speeds_0p25)
plt.plot(spike_rates_0p25[speed_corr_neurons_index, :].T / 200)
