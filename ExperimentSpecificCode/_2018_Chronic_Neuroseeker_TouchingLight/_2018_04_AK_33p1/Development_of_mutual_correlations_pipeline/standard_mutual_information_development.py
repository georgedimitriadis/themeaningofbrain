
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from npeet.lnc import MI

import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl

from BrainDataAnalysis import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

#  -------------------------------------------------
#  GET FOLDERS
#  -------------------------------------------------
dlc_folder = r'D:\Data\George\AK_33.1\2018_04_30-11_38\Analysis\Deeplabcut'
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Denoised',
                     'Kilosort')
mutual_information_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis',
                                 'Results', 'MutualInformation')

num_of_frames_to_average = 0.25/(1/120)
#  -------------------------------------------------

#  -------------------------------------------------
# CREATE THE MUTUAL INFORMATION MEASURE BETWEEN SPEED AND THE FIRING RATES OF ALL NEURONS (FOR THE WHOLE RECORDING)
#  -------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)

# Use body position to create velocities (both linear and polar)
conversion_const = const.PIXEL_PER_FRAME_TO_METERS_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                         180 * (1/np.pi) * np.arctan2(body_velocities[:, 1], body_velocities[:, 0])]).transpose()

# Get the 250ms averages of firing rates for all neurons and of the speed (body_velocity_polar[0] of the animal
video_frame_spike_rates_filename = join(spikes_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
spike_rates_0p25 = np.load(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'))

speeds_0p25 = binning.rolling_window_with_step(body_velocities_polar[:, 0], np.mean,
                                               num_of_frames_to_average, num_of_frames_to_average)

# Calculate the mutual information between the speed and all firing rates (for the whole of the experiment)
'''
n = 0
mutual_infos_spikes_vs_speed = []
for rate in spike_rates_0p25:
    mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate.tolist()[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND], 
                                                                 speeds_0p25[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND]],
                                        k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))


mutual_infos_spikes_vs_speed = np.array(mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'), mutual_infos_spikes_vs_speed)
'''


template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))
mutual_infos_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'))

speed_very_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > 0.15))
speed_very_corr_neurons = template_info.loc[speed_very_corr_neurons_index]

brain_positions_of_corr_neurons = speed_very_corr_neurons['position Y'].values * const.POSITION_MULT

plt.plot(np.array(spike_rates_0p25[speed_very_corr_neurons, :]).T)

# Calculate MI between the FRs of the two neurons
t1 = MI.mi_LNC([spike_rates_0p25[speed_very_corr_neurons[0], :].tolist(),
                spike_rates_0p25[speed_very_corr_neurons[1], :].tolist()],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

# Calculate MI between the FR of the same neuron and itself
t2 = MI.mi_LNC([spike_rates_0p25[speed_very_corr_neurons[1], :].tolist(),
                spike_rates_0p25[speed_very_corr_neurons[1], :].tolist()],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

# Calculate MI between the speed and itselff
s = MI.mi_LNC([speeds_0p25, speeds_0p25],
               k=10, base=np.exp(1), alpha=0.4, intens=1e-10)

plt.plot(speeds_0p25)
plt.plot(spike_rates_0p25[speed_very_corr_neurons_index, :].T / 200)

plt.plot(binning.rolling_window_with_step(speeds_0p25, np.mean, 100, 1))
plt.plot(binning.rolling_window_with_step(spike_rates_0p25[speed_very_corr_neurons_index[0], :].T / 200, np.mean, 100, 1))

# Shuffle the spike rates of one of the best correlated neurons and calculate MI between the shuffled frs and speed
# (1000 times) to generate the basic chance base of MI
'''
speed_corr_neurons_index = [522]
shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_0p25[speed_corr_neurons_index[0], :const.BRAIN_DATA_UP_TO_QUARTER_SECOND],
                                                 speeds_0p25[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND], 
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4, 
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'), shuffled)
'''

# Have a look at the MIs vs the chance level
shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]
# Log
plt.hist(mutual_infos_spikes_vs_speed, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
# Linear
plt.hist(mutual_infos_spikes_vs_speed, bins= 200)
plt.hist(shuffled, bins=50, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)


speed_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > mean_sh+confi_intervals[1]))
speed_corr_neurons = template_info.loc[speed_corr_neurons_index]

brain_positions_of_corr_neurons = np.array([speed_corr_neurons['position X'].values * const.POSITION_MULT,
                                            speed_corr_neurons['position Y'].values * const.POSITION_MULT])

plt.scatter(brain_positions_of_corr_neurons[0, :], brain_positions_of_corr_neurons[1, :],
            s=mutual_infos_spikes_vs_speed[speed_corr_neurons_index]* 200)


#  -------------------------------------------------
# CREATE THE MUTUAL INFORMATION MEASURE BETWEEN POSITION AND THE FIRING RATES OF ALL NEURONS (FOR THE WHOLE RECORDING)
#  -------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)