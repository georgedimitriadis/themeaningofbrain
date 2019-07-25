
from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis import binning

from npeet.lnc import MI

import pandas as pd
import matplotlib.pyplot as plt

import sequence_viewer as sv
import drop_down as dd
import slider as sl


# -------------------------------------------------
# <editor-fold desc="LOAD FOLDERS AND DATA">
# Folder definitions
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
events_folder = join(data_folder, "events")

analysis_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis')
kilosort_folder = join(analysis_folder, 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

# Load data
events_successful_trial_pokes = np.load(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy'))

events_touch_ball = np.load(join(events_definitions_folder, 'events_touch_ball.npy'))
events_touch_ball_successful_trial = np.load(join(events_definitions_folder, 'events_touch_ball_successful_trial.npy'))

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)
spike_rates_0p25 = np.load(spike_rates_per_250ms_filename)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="GENERATE THE SPEED VECTOR">
# Load the cleaned body positions
body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))

# Use body position to create velocities (both linear and polar)
conversion_const = const.PIXEL_PER_FRAME_TO_CM_PER_SECOND
body_velocities = np.diff(body_positions, axis=0) * conversion_const
body_velocities_polar = np.array([np.sqrt(np.power(body_velocities[:, 0], 2) + np.power(body_velocities[:, 1], 2)),
                                 180 * (1/np.pi) * np.arctan2(body_velocities[:, 1], body_velocities[:, 0])]).transpose()

# Correct the speeds by removing the very large spikes (replace them with the average speeds around them)
speeds = body_velocities_polar[:, 0].copy()
acc = np.diff(speeds)
for i in np.arange(len(acc)):
    if speeds[i] > 120:
        speeds[i] = np.mean(speeds[i-10:i+10])

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="GENERATE THE WINDOWS FOR PATTERNED VS NON-PATTERNED BEHAVIOUR">

# Patterned behaviour is defined as going to the ball and then requiring a short time interval to go to the poke with
# no or very little digression from the "touch ball -> go to poke" pattern
time_between_tb_and_p = 15
frames_between_tb_and_p = time_between_tb_and_p * 120
frames_before_tb = 1 * 120

index_of_events_tb_followed_by_fast_p = np.squeeze(np.argwhere((events_successful_trial_pokes -
                                                                events_touch_ball_successful_trial) /
                                           const.SAMPLING_FREQUENCY < time_between_tb_and_p))

frames_tb_followed_by_fast_p = sync_funcs.time_point_to_frame_from_video_df(ev_video,
                                                                            events_touch_ball_successful_trial[
                                                                                index_of_events_tb_followed_by_fast_p])
frames_fast_p = sync_funcs.time_point_to_frame_from_video_df(ev_video,
                                                             events_successful_trial_pokes[
                                                                index_of_events_tb_followed_by_fast_p])

number_of_patterned_events = len(index_of_events_tb_followed_by_fast_p)


windows_of_patterned_behaviour = [np.arange(frames_tb_followed_by_fast_p[frame],
                                            frames_fast_p[frame], 1) for
                                  frame in np.arange(number_of_patterned_events)]


# <editor-fold desc="Have a look at the selected patterned behaviour events">
frame = 0
ev_video = event_dataframes['ev_video']
events_tb_followed_by_fast_p = events_touch_ball_successful_trial[index_of_events_tb_followed_by_fast_p]


def tp_to_frame(frame):
    return sync_funcs.time_point_to_frame_from_video_df(ev_video, frame)[0] - frames_before_tb


dd.connect_repl_var(globals(), 'events_tb_followed_by_fast_p', 'frame', 'tp_to_frame')
sv.image_sequence(globals(), 'frame', 'video_file')
# </editor-fold>

# Non-patterned behaviour is defined as general exploratory behaviour with no specific repetitive pattern
# We find this behaviour when the rat is far away from both the ball and the poke


#   Look for periods of time where the rat is far away from the ball and the poke (assuming exploring)
minimum_distance_of_rat_to_poke_or_ball = 0.5 # That means 25 cm vertical and 30 cm horizontal
ball_positions_df = event_dataframes['ev_ball_tracking']
body_positions_normalised = 2 * body_positions / 640 - 1


# <editor-fold desc="Create the ball_and_rat_positions_in_frame dataframe (Do Not Run Again)">
ball_and_rat_positions_in_frame = pd.DataFrame(columns=['frame', 'AmpTimePoint', 'RatX', 'RatY', 'BallX', 'BallY'])

i = 0
for _, ball in ball_positions_df.iterrows():
    if i == 0:
        nearest_frame = (np.abs(ball['AmpTimePoints'] - ev_video['AmpTimePoints'])).idxmin()
    else:
        nearest_frame = (np.abs(ball['AmpTimePoints'] - ev_video['AmpTimePoints'].iloc[nearest_frame-10:
                                                                                       nearest_frame+7200])).idxmin()
        ball_and_rat_positions_in_frame = ball_and_rat_positions_in_frame.append(
            {'frame': nearest_frame,
             'AmpTimePoint': ev_video['AmpTimePoints'].iloc[nearest_frame],
             'RatX': body_positions_normalised[nearest_frame, 0],
             'RatY': body_positions_normalised[nearest_frame, 1],
             'BallX': ball['X'], 'BallY': ball['Y']}, ignore_index=True)
    i += 1
    if i % 1000 == 0:
        print(i)

pd.to_pickle(ball_and_rat_positions_in_frame, join(events_folder, 'ball_and_rat_positions_in_frame.df'))
# </editor-fold>

lengths_of_windows_of_patterned_behaviour = [len(windows_of_patterned_behaviour[i])
                                             for i in np.arange(number_of_patterned_events)]

ball_and_rat_positions_in_frame = np.load(join(events_folder, 'ball_and_rat_positions_in_frame.df'), allow_pickle=True)

#   Get the distances between rat and ball and rat and poke
poke_position = [0.9, 0]

distances_rat_to_ball = np.sqrt(
    np.power(ball_and_rat_positions_in_frame['RatX'] - ball_and_rat_positions_in_frame['BallX'], 2) +
    np.power(ball_and_rat_positions_in_frame['RatY'] - ball_and_rat_positions_in_frame['BallY'], 2)).values

distances_rat_to_poke = np.sqrt(
    np.power(ball_and_rat_positions_in_frame['RatX'] - poke_position[0], 2) +
    np.power(ball_and_rat_positions_in_frame['RatY'] - poke_position[1], 2)).values

index_of_frames_of_non_patterned_behaviour = np.squeeze(np.argwhere(np.logical_and(
    distances_rat_to_poke > minimum_distance_of_rat_to_poke_or_ball,
    distances_rat_to_ball >minimum_distance_of_rat_to_poke_or_ball)))

frames_of_non_patterned_behaviour = ball_and_rat_positions_in_frame['frame'].\
                                    iloc[index_of_frames_of_non_patterned_behaviour].values

#   Find the longest, continuous stretches of frames where the rat was both away from the poke and the ball
continous_non_patterned_behaviour_frame_ranges = \
    np.split(frames_of_non_patterned_behaviour, np.squeeze(np.argwhere(np.diff(frames_of_non_patterned_behaviour) > 2)
                                                           + 1))
long_continous_non_patterned_behaviour_frame_ranges = [k for k in continous_non_patterned_behaviour_frame_ranges
                                                       if len(k) > 1000]

lengths_of_non_patterned_behaviour_frame_ranges = [len(long_continous_non_patterned_behaviour_frame_ranges[j]) for j in
                                                   np.arange(len(long_continous_non_patterned_behaviour_frame_ranges))]

#   There are more (just about) continuous stretches than patterned behaviour trials so for each length of frames of a
#   patterned behaviour trial pick the same length of frames from a different continuous stretch of non patterned
#   behaviour (making sure the non patterned stretch has enough frames in it).

#   THE FOLLOWING CODE WILL WORK ONLY IF THE FOLLOWING LINE RETURNS FALSE
print(np.any(np.sort(lengths_of_windows_of_patterned_behaviour) -
             np.sort(lengths_of_non_patterned_behaviour_frame_ranges[:number_of_patterned_events]) > 0))

windows_of_non_patterned_behaviour = []
for index_of_trial in np.arange(number_of_patterned_events):
    length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[index_of_trial]
    index_of_frame_range_of_non_patterned_behaviour = np.argsort(lengths_of_non_patterned_behaviour_frame_ranges)[index_of_trial]
    frame_range_of_non_patterned_behaviour = long_continous_non_patterned_behaviour_frame_ranges[
        index_of_frame_range_of_non_patterned_behaviour]
    start_point_value = int(np.random.choice(frame_range_of_non_patterned_behaviour[: - length_of_trial]))
    windows_of_non_patterned_behaviour.append(np.arange(start_point_value, start_point_value + length_of_trial, 1))

lengths_of_windows_of_non_patterned_behaviour = [len(windows_of_non_patterned_behaviour[i])
                                                 for i in np.arange(number_of_patterned_events)]

'''
# Code for just in case
windows_of_patterned_behaviour_flat = np.array([windows_of_patterned_behaviour[trial][frame]
                                               for trial in np.arange(number_of_patterned_events)
                                               for frame in np.arange(len(windows_of_patterned_behaviour[trial]))])

windows_of_non_patterned_behaviour_flat = np.array([windows_of_non_patterned_behaviour[trial][frame]
                                                   for trial in np.arange(number_of_patterned_events)
                                                   for frame in np.arange(len(windows_of_non_patterned_behaviour[trial]))])
'''
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="USING THE GENERATED WINDOWS SMOOTH THE CORRESPONDING PARTS OF THE SPEED AND THE FIRING RATES">

#   Smooth speeds and spike rates for each patterned behaviour
speeds_patterned_behaviour_0p25 = [s for k in np.arange(number_of_patterned_events) for s in
                                   binning.rolling_window_with_step(speeds[windows_of_patterned_behaviour[k]],
                                                                    np.mean, 30, 30)]
spike_rates_patterned_behaviour_0p25 = np.empty(1)
for k in np.arange(number_of_patterned_events):
    smoothed_data = binning.rolling_window_with_step(spike_rates[:, windows_of_patterned_behaviour[k]], np.mean, 30, 30)
    smoothed_data = np.array(smoothed_data)
    if k == 0:
        spike_rates_patterned_behaviour_0p25 = smoothed_data
    else:
        spike_rates_patterned_behaviour_0p25 = np.concatenate((spike_rates_patterned_behaviour_0p25, smoothed_data), axis=1)


#   Smooth speeds and spike rates for each non patterned trial
speeds_non_patterned_behaviour_0p25 = [s for k in np.arange(number_of_patterned_events) for s in
                                       binning.rolling_window_with_step(speeds[windows_of_non_patterned_behaviour[k]],
                                                                        np.mean, 30, 30)]
spike_rates_non_patterned_behaviour_0p25 = np.empty(1)
for k in np.arange(number_of_patterned_events):
    smoothed_data = binning.rolling_window_with_step(spike_rates[:, windows_of_non_patterned_behaviour[k]], np.mean, 30, 30)
    smoothed_data = np.array(smoothed_data)
    if k == 0:
        spike_rates_non_patterned_behaviour_0p25 = smoothed_data
    else:
        spike_rates_non_patterned_behaviour_0p25 = np.concatenate((spike_rates_non_patterned_behaviour_0p25,
                                                                   smoothed_data), axis=1)

#   Check that the two speed distributions are roughly the same
_ = plt.hist(speeds_non_patterned_behaviour_0p25)
_ = plt.hist(speeds_patterned_behaviour_0p25)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="RUN THE MUTUAL INFORMATION">

#   Do the patterned behaviour
n = 0
mi_pb_spikes_vs_speed = []
for rate in spike_rates_patterned_behaviour_0p25:
    mi_pb_spikes_vs_speed.append(MI.mi_LNC([rate.tolist(), speeds_patterned_behaviour_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_pb_spikes_vs_speed = np.array(mi_pb_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'), mi_pb_spikes_vs_speed)

#   Do the shuffle of the best patterned behaviour
max_neuron = np.argmax(mi_pb_spikes_vs_speed)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_patterned_behaviour_0p25[max_neuron],
                                                 speeds_patterned_behaviour_0p25,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_589_vs_speed_patterned_behaviour.npy'), shuffled)

#   Do the non patterned behaviour
n = 0
mi_non_pb_spikes_vs_speed = []
for rate in spike_rates_non_patterned_behaviour_0p25:
    mi_non_pb_spikes_vs_speed.append(MI.mi_LNC([rate.tolist(), speeds_non_patterned_behaviour_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_non_pb_spikes_vs_speed = np.array(mi_non_pb_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_non_patterned_behaviour.npy'),
        mi_non_pb_spikes_vs_speed)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS">
mi_pb_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'))
mi_non_pb_spikes_vs_speed = np.load(join(mutual_information_folder,
                                         'mutual_infos_spikes_vs_speed_non_patterned_behaviour.npy'))


# Have a look at the MIs vs the chance level
shuffled = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'))
mean_sh = np.mean(shuffled)
confidence_level = 0.99
confi_intervals = shuffled[int((1. - confidence_level) / 2 * 1000)], shuffled[int((1. + confidence_level) / 2 * 1000)]
# Log
plt.hist(mi_pb_spikes_vs_speed, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(mi_non_pb_spikes_vs_speed, bins=np.logspace(np.log10(0.0001), np.log10(1), 50))
plt.hist(shuffled, bins=np.logspace(np.log10(0.0001), np.log10(1), 50), color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 60)
plt.gca().set_xscale("log")
# Linear
plt.hist(mi_pb_spikes_vs_speed, bins= 200)
plt.hist(mi_non_pb_spikes_vs_speed, bins= 200)
plt.hist(shuffled, bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 20)

speed_corr_neurons_pb_index = np.squeeze(np.argwhere(mi_pb_spikes_vs_speed > mean_sh+confi_intervals[1]))
speed_corr_neurons_pb = template_info.loc[speed_corr_neurons_pb_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=speed_corr_neurons_pb,
                                     dot_sizes=mi_pb_spikes_vs_speed[speed_corr_neurons_pb_index] * 4000,
                                     font_size=5)
speed_corr_neurons_non_pb_index = np.squeeze(np.argwhere(mi_non_pb_spikes_vs_speed > mean_sh+confi_intervals[1]))
speed_corr_neurons_non_pb = template_info.loc[speed_corr_neurons_non_pb_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=speed_corr_neurons_non_pb,
                                     dot_sizes=mi_pb_spikes_vs_speed[speed_corr_neurons_non_pb_index] * 4000,
                                     font_size=5)


def draw1(index):
    plt.clf()
    plt.plot(speeds_patterned_behaviour_0p25)
    plt.plot(spike_rates_patterned_behaviour_0p25[speed_corr_neurons_pb_index[index], :])
    return None


index = 0
out = None
sl.connect_repl_var(globals(), 'index', 'out', 'draw1', slider_limits=[0, len(speed_corr_neurons_pb_index) - 1])


def draw2(index):
    plt.clf()
    plt.plot(speeds_patterned_behaviour_0p25)
    plt.plot(speeds_non_patterned_behaviour_0p25)
    plt.plot(spike_rates_non_patterned_behaviour_0p25[index, :])
    plt.plot(spike_rates_patterned_behaviour_0p25[index, :])
    return None

index = 0
out = None
sl.connect_repl_var(globals(), 'index', 'out', 'draw2', slider_limits=[0, len(spike_rates_patterned_behaviour_0p25) - 1])


# </editor-fold>
# -------------------------------------------------

# THERE IS SOMETHING HAPPENING AT 328 INDEX OF THE PATTERNED WINDOWS! 