


from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from npeet.lnc import MI

import one_shot_viewer as osv
import sequence_viewer as sv
import transform as tr
import slider as sl
import common_data_transforms as cdt

from BrainDataAnalysis import binning
from BehaviorAnalysis import dlc_post_processing as dlc_pp

from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs


#  -------------------------------------------------
#  GET FOLDERS
#  -------------------------------------------------
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')

spikes_folder = join(analysis_folder, 'Denoised', 'Kilosort')
mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')
poke_folder = join(analysis_folder, 'Results', 'EventsCorrelations', 'Poke')
events_folder = join(data_folder, "events")

num_of_frames_to_average = 0.25/(1/120)

template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(spikes_folder, 'spike_info_after_cortex_sorting.df'))

lfp_downsample_factor = 4  # possible 3, 4, 5, 10
ds_numpy_filename = join(analysis_folder, 'Lfp', 'Downsampling', 'Amplifier_LFPs_Downsampled_x{}.npy'.format(lfp_downsample_factor))
downsampled_lfp = np.load(ds_numpy_filename)
#  -------------------------------------------------

#  -------------------------------------------------
#  GENERATE THE SPEED VECTOR
#  -------------------------------------------------

# Load the clean body markers
updated_markers_filename = join(dlc_project_folder, 'post_processing', 'cleaned_body_marker_positions_order_{}_gap_{}.df'.
                                format(str(4), str(10)))
updated_body_markers_positions = pd.read_pickle(updated_markers_filename)

# Use the markers to create a single body position
body_positions = dlc_pp.average_multiple_markers_to_single_one(updated_body_markers_positions, flip=True)
body_positions = dlc_pp.clean_large_movements_single_axis(body_positions, maximum_pixels=12)

# Use body position to create velocities (both linear and polar)
conversion_const = const.PIXEL_PER_FRAME_TO_CM_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                         180 * (1/np.pi) * np.arctan2(body_velocities[:, 1], body_velocities[:, 0])]).transpose()

# Correct the speeds by removing the very large spikes (replace them with the average speeds around them)
speeds = body_velocities_polar[:, 0].copy()
acc = np.diff(speeds)
for i in np.arange(len(acc)):
    if acc[i] > 200:
        speeds[i+1] = (speeds[i] + speeds[i+2])/2

# Get the 250ms averages of firing rates for all neurons and of the speed (body_velocity_polar[0] of the animal
video_frame_spike_rates_filename = join(spikes_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates = np.load(video_frame_spike_rates_filename)
spike_rates_0p25 = np.load(join(spikes_folder, 'firing_rate_with_0p25s_window.npy'))

speeds_0p25 = binning.rolling_window_with_step(speeds, np.mean,
                                               num_of_frames_to_average, num_of_frames_to_average)
speeds_0p25_array = np.array(speeds_0p25)


#  -------------------------------------------------
# CREATE THE MUTUAL INFORMATION MEASURE BETWEEN SPEED AND THE FIRING RATES OF ALL NEURONS (FOR THE WHOLE RECORDING)
#  -------------------------------------------------

# Calculate the mutual information between the speed and all firing rates (for the whole of the experiment)
# using the lnc code (a Kraskov with some local non-uniform correction for better very high correlations)

n = 0
mutual_infos_spikes_vs_speed = []
for rate in spike_rates_0p25:
    mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate.tolist()[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND], 
                                                                 speeds_0p25[:const.BRAIN_DATA_UP_TO_QUARTER_SECOND]],
                                        k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))


mutual_infos_spikes_vs_speed = np.array(mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected_for_large_acc.npy'), mutual_infos_spikes_vs_speed)


movement_bins = np.squeeze(np.argwhere(speeds_0p25_array > 4))
moving_spike_rates_0p25 = np.squeeze(spike_rates_0p25[:, movement_bins])
moving_speeds_0p25 = speeds_0p25_array[movement_bins].tolist()

end_of_data = int(np.squeeze(np.argwhere(movement_bins == const.BRAIN_DATA_UP_TO_QUARTER_SECOND)))
n = 0
mutual_infos_spikes_vs_speed = []
for rate in moving_spike_rates_0p25:
    mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate.tolist()[:end_of_data],
                                                   moving_speeds_0p25[:end_of_data]],
                                        k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))


mutual_infos_spikes_vs_speed = np.array(mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected_for_large_acc_moving_only.npy'), mutual_infos_spikes_vs_speed)


mutual_infos_spikes_vs_speed_original = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed.npy'))
# OR this after I have corrected the speed to remove the huge speed change spikes (due to markers disappearing)
mutual_infos_spikes_vs_speed_corrected = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected_for_large_acc.npy'))
# OR this after I have removed the parts where the animal is sitting still
mutual_infos_spikes_vs_speed_corrected_only_mov = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected_for_large_acc_moving_only.npy'))

_ = plt.hist(mutual_infos_spikes_vs_speed_corrected, 200)
_ = plt.hist(mutual_infos_spikes_vs_speed_corrected_only_mov, 200)

speed_very_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed_corrected > 0.15))
speed_very_corr_neurons = template_info.iloc[speed_very_corr_neurons_index]

brain_positions_of_corr_neurons = speed_very_corr_neurons['position Y'].values * const.POSITION_MULT

largest_cor_neuron_index = 522
plt.plot(np.array(spike_rates_0p25[largest_cor_neuron_index, :]).T)


plt.plot(speeds_0p25)
plt.plot(spike_rates_0p25[largest_cor_neuron_index, :].T)

plt.plot(speeds)
plt.plot(spike_rates[largest_cor_neuron_index, :].T)

plt.plot(binning.rolling_window_with_step(speeds_0p25, np.mean, 100, 1))
plt.plot(binning.rolling_window_with_step(spike_rates_0p25[speed_very_corr_neurons_index[0], :].T / 200, np.mean, 100, 1))

s_20 = np.array(binning.rolling_window_with_step(speeds_0p25, np.mean, 8, 1))
plt.plot((s_20 - s_20.mean()) /s_20.std())
f_20 = np.array(binning.rolling_window_with_step(spike_rates_0p25[speed_very_corr_neurons_index, :].T * 2, np.mean, 8, 1))
plt.plot((f_20 - f_20.mean()) / f_20.std())

# Shuffle the spike rates of one of the best correlated neurons and calculate MI between the shuffled frs and speed
# (1000 times) to generate the basic chance base of MI
'''
speed_corr_neurons_index = [largest_cor_neuron_index]
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
plt.hist(mutual_infos_spikes_vs_speed_corrected_only_mov, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
# Linear
plt.hist(mutual_infos_spikes_vs_speed_corrected_only_mov, bins= 200)
plt.hist(shuffled, bins=50, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)


speed_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed_corrected > mean_sh+confi_intervals[1]))
speed_corr_neurons = template_info.loc[speed_corr_neurons_index]

brain_positions_of_corr_neurons = np.array([speed_corr_neurons['position X'].values * const.POSITION_MULT,
                                            speed_corr_neurons['position Y'].values * const.POSITION_MULT])

spp.view_grouped_templates_positions(spikes_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=speed_corr_neurons,
                                     dot_sizes=mutual_infos_spikes_vs_speed_corrected[speed_corr_neurons_index] * 4000,
                                     font_size=5)


#  -------------------------------------------------
# DO A TIME SHIFT WITH THW BEST NEURON TO SEE IF THERE ARE TIME DEPENDENCIES
#  -------------------------------------------------
largest_cor_neuron_index = 522
# Average the speed and the firing rate but with a moving window with step one. Then randomly select 10 windows of 3000
# points each (to keep length of arrays low)
rate_base = np.squeeze(spike_rates[largest_cor_neuron_index, :])
rate_base_moving_window_averaged = binning.rolling_window_with_step(rate_base, np.mean, 70, 1)
speed_moving_window_averaged = binning.rolling_window_with_step(speeds, np.mean, 70, 1)

'''
windows_to_pick = []
window_size = 3000
for i in np.arange(1, 11, 1):
    range_to_pick = np.arange((len(rate_base)/10 - 30000) * i, (len(rate_base)/10 - window_size) * i, 1)
    starting_point = np.random.choice(range_to_pick, 1)
    windows_to_pick.append(np.arange(starting_point, starting_point+window_size, 1))

windows_to_pick = np.array(windows_to_pick).flatten().astype(np.int)
np.save(join(mutual_information_folder, 'random_windows_of_indices_for_shifting_mi.npy'), windows_to_pick)
'''

windows_to_pick = np.load(join(mutual_information_folder, 'random_windows_of_indices_for_shifting_mi.npy'))
rate_base_moving_window_averaged = np.array(rate_base_moving_window_averaged)[windows_to_pick].tolist()
speed_moving_window_averaged = np.array(speed_moving_window_averaged)[windows_to_pick].tolist()

total_shift = 2*100

n = 0
time_shifted_mutual_infos_spikes_vs_speed = []
for shift in np.arange(total_shift):
    rate = rate_base_moving_window_averaged[shift:-(total_shift-shift)]
    time_shifted_mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate,
                                                                speed_moving_window_averaged[int(total_shift/2):-int(total_shift/2)]],
                                                     k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done shift {}'.format(str(n)))


time_shifted_mutual_infos_spikes_vs_speed = np.array(time_shifted_mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder,
             'mutual_infos_spikes_vs_speed_corrected_for_large_acc_moving_only_time_shifted_70framesavg.npy'),
        time_shifted_mutual_infos_spikes_vs_speed)


time_shifted_mutual_infos_spikes_vs_speed = \
    np.load(join(mutual_information_folder,
                 'mutual_infos_spikes_vs_speed_corrected_for_large_acc_moving_only_time_shifted_80framesavg.npy'))

plt.plot(np.arange(-100*8.33, 100*8.33, 8.33), time_shifted_mutual_infos_spikes_vs_speed)
plt.vlines(x=np.argmax(time_shifted_mutual_infos_spikes_vs_speed)*8.33 - 100*8.33,
           ymin=time_shifted_mutual_infos_spikes_vs_speed.min(),
           ymax=time_shifted_mutual_infos_spikes_vs_speed.max())
plt.title('Firing rate predicting speed')

# Look at all the shifted MI generated with different sizes of smoothing windows through the data
labels = []
for n in [5, 20, 50, 70]:
    t = np.load(join(mutual_information_folder,
                     'mutual_infos_spikes_vs_speed_corrected_for_large_acc_moving_only_time_shifted_{}framesavg.npy').format(str(n)))
    t = (t - t.min()) / (t.max() - t.min())
    label, = plt.plot(np.arange(-100*8.33, 100*8.33, 8.33), t, label='Window size = {} milliseconds'.format(str(n * 8.33)))
    labels.append(label)
plt.legend(handles=labels)

#  -------------------------------------------------
# DO A TIME SHIFT WITH ALL CORRELATED NEURONS TO SEE IF THERE ARE TIME DEPENDENCIES
#  -------------------------------------------------

windows_to_pick = np.load(join(mutual_information_folder, 'random_windows_of_indices_for_shifting_mi.npy'))

total_shift = 2*100
speed_moving_window_averaged = binning.rolling_window_with_step(speeds, np.mean, 50, 1)
speed_moving_window_averaged = np.array(speed_moving_window_averaged)[windows_to_pick].tolist()

n = 1
for neuron_index in speed_corr_neurons_index:
    neuron_template = template_info['template number'].loc[neuron_index]
    rate_base = np.squeeze(spike_rates[neuron_index, :])
    rate_base_moving_window_averaged = binning.rolling_window_with_step(rate_base, np.mean, 50, 1)
    rate_base_moving_window_averaged = np.array(rate_base_moving_window_averaged)[windows_to_pick].tolist()

    s = 0
    time_shifted_mutual_infos_spikes_vs_speed = []
    for shift in np.arange(total_shift):
        rate = rate_base_moving_window_averaged[shift:-(total_shift-shift)]
        time_shifted_mutual_infos_spikes_vs_speed.append(MI.mi_LNC([rate,
                                                                    speed_moving_window_averaged[int(total_shift/2):-int(total_shift/2)]],
                                                         k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
        s += 1
        print('Done shift {} for neuron {}'.format(str(s), str(n)))

    time_shifted_mutual_infos_spikes_vs_speed = np.array(time_shifted_mutual_infos_spikes_vs_speed)
    np.save(join(mutual_information_folder,
                 'mutual_infos_spikes_of_template_{}_vs_speed_corrected_for_large_acc_time_shifted_50framesavg.npy'.
                 format(str(n))),
            time_shifted_mutual_infos_spikes_vs_speed)
    n += 1

#  -------------------------------------------------
# CHECK IF SPEED SHIFTED CORRELATES WITH ITSELF
#  -------------------------------------------------

windows_to_pick = np.load(join(mutual_information_folder, 'random_windows_of_indices_for_shifting_mi.npy'))
speed_moving_window_averaged = np.array(speed_moving_window_averaged)[windows_to_pick].tolist()

total_shift = 2*100

n = 0
time_shifted_mutual_infos_speed_vs_speed = []
for shift in np.arange(total_shift):
    shifted_speed = speed_moving_window_averaged[shift:-(total_shift-shift)]
    time_shifted_mutual_infos_speed_vs_speed.append(MI.mi_LNC([shifted_speed,
                                                                speed_moving_window_averaged[int(total_shift/2):-int(total_shift/2)]],
                                                     k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done shift {}'.format(str(n)))


time_shifted_mutual_infos_speed_vs_speed = np.array(time_shifted_mutual_infos_speed_vs_speed)
np.save(join(mutual_information_folder,
             'mutual_infos_speed_vs_speed_corrected_for_large_acc_moving_only_time_shifted_50framesavg.npy'),
        time_shifted_mutual_infos_speed_vs_speed)


#  -------------------------------------------------
# QUICK CHECK FOR LFP PHASE ALIGNMENT
#  -------------------------------------------------

largest_cor_neuron_index = 522
largest_cor_neuron = template_info.iloc[largest_cor_neuron_index]
largest_cor_neuron_spike_times = spike_info['times'].loc[spike_info['template_after_sorting'] ==
                                                         largest_cor_neuron['template number']].values / lfp_downsample_factor
largest_cor_neuron_spike_times = largest_cor_neuron_spike_times.astype(np.int)

window_size = int(0.5 * (const.SAMPLING_FREQUENCY / lfp_downsample_factor))

spike_triggered_lfps = []
for spike in largest_cor_neuron_spike_times:
    spike_triggered_lfps.append(downsampled_lfp[:, spike - window_size:spike + window_size])
spike_triggered_lfps = np.array(spike_triggered_lfps)
spike_triggered_lfps_mean = spike_triggered_lfps.mean(axis=0)

random_times = np.random.choice(np.arange(2*window_size,
                                          downsampled_lfp.shape[1]-2*window_size,
                                          1),
                                len(largest_cor_neuron_spike_times))
random_triggered_lfps = []
for spike in random_times:
    random_triggered_lfps.append(downsampled_lfp[:, spike - window_size:spike + window_size])
random_triggered_lfps = np.array(random_triggered_lfps)
random_triggered_lfps = random_triggered_lfps.mean(axis=0)

z_factor = []
for channel in np.arange(spike_triggered_lfps_mean.shape[0]):
    z_factor.append((spike_triggered_lfps_mean[channel, :] - random_triggered_lfps[channel, :].mean()) /
                    random_triggered_lfps[channel, :].std())

z_factor = np.array(z_factor)

plt.plot(np.arange(-window_size/const.SAMPLING_FREQUENCY, window_size/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY),
                   cdt.space_data(z_factor, 10).T)

plt.plot(np.arange(-window_size/const.SAMPLING_FREQUENCY, window_size/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY),
         z_factor[28:35, :].T)

lfp_probe_positions = np.empty(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)
lfp_probe_positions[np.arange(0, 72, 2)] = (((np.arange(9, 1440, 40) + 1) / 4).astype(np.int) + 1) * 22.5
lfp_probe_positions[np.arange(1, 72, 2)] = (((np.arange(29, 1440, 40) + 1) / 4).astype(np.int)) * 22.5

z_factor_positioned = z_factor * 10 + np.tile(lfp_probe_positions, (z_factor.shape[1], 1)).T

plt.plot(np.arange(-window_size/const.SAMPLING_FREQUENCY, window_size/const.SAMPLING_FREQUENCY, 1/const.SAMPLING_FREQUENCY),
         z_factor_positioned.T)
plt.hlines(y=largest_cor_neuron['position Y'] * const.POSITION_MULT,
           xmax=window_size/const.SAMPLING_FREQUENCY, xmin=-window_size/const.SAMPLING_FREQUENCY)


# Look at the time shifted  MI of all correlated neurons
labels = []
ims = []
maxima = []
for n in range(41):
    t = np.load(join(mutual_information_folder,
                     'mutual_infos_spikes_of_template_{}_vs_speed_corrected_for_large_acc_time_shifted_50framesavg.npy').format(str(n)))
    ims.append(t)
    t_n = (t - t.min()) / (t.max() - t.min())
    t_s = binning.rolling_window_with_step(t, np.mean, 1, 1)
    if t[100] > 0.01:
        label, = plt.plot(np.arange(-100*8.33, 100*8.33, 8.33), t, label='Template = {}'.format(str(n)))
        #labels.append(label)
        maxima.append((np.argmax(t_s) - 100) * 8.33)


avg = np.mean(maxima)
var = np.var(maxima)
pdf_x = np.linspace(np.min(maxima),np.max(maxima),100)
pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)
plt.hist(maxima, 20, density=True)
plt.plot(pdf_x, pdf_y, 'k--')
plt.title('Mean = {}'.format(str(avg)))

ims = np.array(ims)
index = 0
x = np.arange(-100*8.33, 100*8.33, 8.33)
sv.graph_pane(globals(), 'index', 'ims', 'x')



#  -------------------------------------------------
# <editor-fold desc=CHECK IF DIFFERENT SETS OF NEURONS ARE CORRELATED WITH SPEED FOR SUCCESSFUL TRIALS VS OTHER PERIODS
time_points_of_trial_pokes = np.load(join(poke_folder, 'time_points_of_trial_pokes.npy'))
time_points_of_non_trial_pokes = np.load(join(poke_folder, 'time_points_of_non_trial_pokes.npy'))
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
time_points_of_touch_ball = event_dataframes['ev_rat_touch_ball']['AmpTimePoints']

camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
all_pokes = beam_breaks[:, 0]

# Get the touch ball time points that are followed by a poke (successful trials)
time_points_touch_ball_followed_by_poke = []
index_p = 0
index_tb = 0
for touch_ball in time_points_of_touch_ball:
    if time_points_of_trial_pokes[index_p] - touch_ball < 40 * const.SAMPLING_FREQUENCY:
        time_points_touch_ball_followed_by_poke.append(touch_ball)
        index_p += 1
time_points_touch_ball_followed_by_poke = np.array(time_points_touch_ball_followed_by_poke)

# Check that none of the non_trial_poke time points are between a touch ball and a trial poke (shouldn't be any)
weird_non_trial_pokes = []
total_time = 0
for nt_poke in time_points_of_non_trial_pokes:
    for poke_after_ball_index in np.arange(len(time_points_touch_ball_followed_by_poke)):
        start = time_points_touch_ball_followed_by_poke[poke_after_ball_index]
        end = time_points_of_trial_pokes[poke_after_ball_index]
        total_time += end - start
        if nt_poke > start and nt_poke < end:
            weird_non_trial_pokes.append(nt_poke)
total_time /= len(time_points_touch_ball_followed_by_poke)

time_points_of_non_trial_pokes = np.delete(time_points_of_non_trial_pokes, np.argwhere(np.isin(time_points_of_non_trial_pokes, weird_non_trial_pokes)))

# Get non_trial_poke time points that are as forwards in time from another poke as the corresponding difference between
# touch ball and successful poke
non_trial_pokes_in_all_pokes = np.squeeze(np.argwhere(np.isin(all_pokes, time_points_of_non_trial_pokes)))
dif_of_non_trial_pokes_to_previous_poke = (all_pokes[non_trial_pokes_in_all_pokes] -
                                           all_pokes[non_trial_pokes_in_all_pokes - 1])


dif_of_non_trial_pokes_to_previous_poke_to_select_from = np.copy(dif_of_non_trial_pokes_to_previous_poke)
non_trial_pokes_in_all_pokes_to_select_from = np.copy(non_trial_pokes_in_all_pokes)

trial_durations = time_points_of_trial_pokes - time_points_touch_ball_followed_by_poke
order_of_trials_by_decreasing_duration = np.argsort(trial_durations)[::-1]

time_points_of_non_trial_pokes_selected = []
for poke_after_ball_index in order_of_trials_by_decreasing_duration:
    start = time_points_touch_ball_followed_by_poke[poke_after_ball_index]
    end = time_points_of_trial_pokes[poke_after_ball_index]
    duration = end - start

    try:
        selecetd_index = np.argwhere(dif_of_non_trial_pokes_to_previous_poke_to_select_from > duration)[-1]
    except IndexError:  # This is for the few occasions where the non trials do not have such a long duration
        selecetd_index = np.argwhere(dif_of_non_trial_pokes_to_previous_poke_to_select_from > 0.6 * duration)[-1]

    time_points_of_non_trial_pokes_selected.append(
        all_pokes[non_trial_pokes_in_all_pokes_to_select_from[np.squeeze(selecetd_index)]])
    dif_of_non_trial_pokes_to_previous_poke_to_select_from = \
        np.delete(dif_of_non_trial_pokes_to_previous_poke_to_select_from, selecetd_index)
    all_pokes_to_select_from = np.delete(non_trial_pokes_in_all_pokes_to_select_from, selecetd_index)
time_points_of_non_trial_pokes_selected = np.squeeze(np.array(time_points_of_non_trial_pokes_selected))

# Create the corresponding time windows (and in frames)
window_size_frames = np.array(trial_durations / const.SAMPLING_FREQUENCY * 120, dtype=np.int32)
trial_frames = sync_funcs.time_point_to_frame_from_video_df(event_dataframes['ev_video'], time_points_of_trial_pokes)
non_trial_frames = sync_funcs.time_point_to_frame_from_video_df(event_dataframes['ev_video'],
                                                                time_points_of_non_trial_pokes_selected)

number_of_trial_pokes = len(trial_frames)
number_of_non_trial_pokes = len(non_trial_frames)

trial_windows_in_frames = [np.arange(trial_frames[tf] - window_size_frames[tf],
                                     trial_frames[tf] + window_size_frames[tf], 1) for
                           tf in np.arange(len(trial_frames))]
non_trial_windows_in_frames = [np.arange(non_trial_frames[ntf] - window_size_frames[ntf],
                                         non_trial_frames[ntf] + window_size_frames[ntf], 1)
                               for ntf in np.arange(len(non_trial_frames))]

trial_windows_in_frames = list(itertools.chain(*trial_windows_in_frames))
non_trial_windows_in_frames = list(itertools.chain(*non_trial_windows_in_frames))

# Get the neurons correlated with speed and the speeds and then smooth them and cut them
mutual_infos_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_corrected.npy'))
template_info = pd.read_pickle(join(spikes_folder, 'template_info.df'))

shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_522_vs_speed.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]


speed_corr_neurons_index = np.squeeze(np.argwhere(mutual_infos_spikes_vs_speed > mean_sh+confi_intervals[1]))
speed_corr_neurons = template_info.loc[speed_corr_neurons_index]


num_of_frames_to_average_firing_rates = 0.25 * 120

speed_moving_window_averaged = binning.rolling_window_with_step(speeds, np.mean, num_of_frames_to_average_firing_rates,
                                                                1)
trials_speed_moving_window_averaged = np.array(speed_moving_window_averaged)[trial_windows_in_frames].tolist()
non_trials_speed_moving_window_averaged = np.array(speed_moving_window_averaged)[non_trial_windows_in_frames].tolist()

# Run the MI
trial_mutual_infos_spikes_vs_speed = []
non_trial_mutual_infos_spikes_vs_speed = []
n = 0
for neuron_index in speed_corr_neurons_index:
    neuron_template = template_info['template number'].loc[neuron_index]
    rate_base = np.squeeze(spike_rates[neuron_index, :])
    rate_base_moving_window_averaged = binning.rolling_window_with_step(rate_base, np.mean,
                                                                        num_of_frames_to_average_firing_rates,
                                                                        1)

    trials_rate = np.array(rate_base_moving_window_averaged)[trial_windows_in_frames].tolist()
    non_trials_rate = np.array(rate_base_moving_window_averaged)[non_trial_windows_in_frames].tolist()

    trial_mutual_infos_spikes_vs_speed.append(MI.mi_LNC([trials_rate,
                                                         trials_speed_moving_window_averaged],
                                                        k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    non_trial_mutual_infos_spikes_vs_speed.append(MI.mi_LNC([non_trials_rate,
                                                            non_trials_speed_moving_window_averaged],
                                                            k=10, base=np.exp(1), alpha=0.4, intens=1e-10))

    print('Done neuron {}'.format(str(n)))
    n += 1

trial_mutual_infos_spikes_vs_speed = np.array(trial_mutual_infos_spikes_vs_speed)
non_trial_mutual_infos_spikes_vs_speed = np.array(non_trial_mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'trial_mutual_infos_spikes_vs_speed.npy'),
        trial_mutual_infos_spikes_vs_speed)
np.save(join(mutual_information_folder, 'non_trial_mutual_infos_spikes_vs_speed.npy'),
        non_trial_mutual_infos_spikes_vs_speed)


plt.plot(trial_mutual_infos_spikes_vs_speed)
plt.plot(non_trial_mutual_infos_spikes_vs_speed)
plt.plot(mutual_infos_spikes_vs_speed[speed_corr_neurons_index])

plt.scatter(trial_mutual_infos_spikes_vs_speed, mutual_infos_spikes_vs_speed[speed_corr_neurons_index])