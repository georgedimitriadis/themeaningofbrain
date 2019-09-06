
from os.path import join
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs
from BrainDataAnalysis.Spike_Sorting import positions_on_probe as spp
from BrainDataAnalysis import binning

from npeet.lnc import MI
from npeet.entropy_estimators import *

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
kilosort_folder = join(analysis_folder, 'Denoised', 'Kilosort')
results_folder = join(analysis_folder, 'Results')
events_definitions_folder = join(results_folder, 'EventsDefinitions')

mutual_information_folder = join(analysis_folder, 'Results', 'MutualInformation')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-05-07')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

# Load data
events_successful_trial_pokes = np.load(join(events_definitions_folder, 'events_pokes_of_successful_trial.npy'))

events_touch_ball = np.load(join(events_definitions_folder, 'events_touch_ball.npy'))
events_touch_ball_successful_trial = np.load(join(events_definitions_folder, 'events_touch_ball_successful_trial.npy'))

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))

spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cortex_sorting.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)
spike_rates_0p25 = np.load(spike_rates_per_250ms_filename)

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))
# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="GENERATE THE SPEED VECTOR (SAVED. DO NOT RUN AGAIN)">
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

np.save(join(dlc_project_folder, 'post_processing', 'speeds.npy'), speeds)
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

# The events with indices 3,4 and 5 are bonsai bugs where the ball appears and dissaperas immediately without touch
index_of_events_tb_followed_by_fast_p = index_of_events_tb_followed_by_fast_p[3:]

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
minimum_distance_of_rat_to_poke_or_ball = 0.7 # That means 25 cm vertical and 30 cm horizontal
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
             'RatY': -body_positions_normalised[nearest_frame, 1],
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

distances_rat_to_poke_when_ball_in_on = np.sqrt(
    np.power(ball_and_rat_positions_in_frame['RatX'] - poke_position[0], 2) +
    np.power(ball_and_rat_positions_in_frame['RatY'] - poke_position[1], 2)).values

index_of_frames_of_non_patterned_behaviour = np.squeeze(np.argwhere(np.logical_and(
    distances_rat_to_poke_when_ball_in_on > minimum_distance_of_rat_to_poke_or_ball,
    distances_rat_to_ball > minimum_distance_of_rat_to_poke_or_ball)))

frames_of_non_patterned_behaviour = ball_and_rat_positions_in_frame['frame'].\
                                    iloc[index_of_frames_of_non_patterned_behaviour].values

#   Find the longest, continuous stretches of frames where the rat was both away from the poke and the ball
continous_non_patterned_behaviour_frame_ranges = \
    np.split(frames_of_non_patterned_behaviour, np.squeeze(np.argwhere(np.diff(frames_of_non_patterned_behaviour) > 2)
                                                           + 1))
long_continous_non_patterned_behaviour_frame_ranges = [k for k in continous_non_patterned_behaviour_frame_ranges
                                                       if len(k) > 100]

lengths_of_non_patterned_behaviour_frame_ranges = [len(long_continous_non_patterned_behaviour_frame_ranges[j]) for j in
                                                   np.arange(len(long_continous_non_patterned_behaviour_frame_ranges))]


windows_of_non_patterned_behaviour = []
for index_of_trial in np.arange(number_of_patterned_events):
    length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[index_of_trial]
    if index_of_trial < len(lengths_of_non_patterned_behaviour_frame_ranges):
        index_of_frame_range_of_non_patterned_behaviour = np.argsort(lengths_of_non_patterned_behaviour_frame_ranges)[
            index_of_trial]
        frame_range_of_non_patterned_behaviour = long_continous_non_patterned_behaviour_frame_ranges[
            index_of_frame_range_of_non_patterned_behaviour]
        windows_of_non_patterned_behaviour.append(np.arange(frame_range_of_non_patterned_behaviour[0],
                                                            frame_range_of_non_patterned_behaviour[0] + length_of_trial,
                                                            1))
    else:
        start = windows_of_non_patterned_behaviour[-1][-1]
        windows_of_non_patterned_behaviour.append(np.arange(start+1, start + 1 + length_of_trial, 1))

length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[10]
windows_of_non_patterned_behaviour[10] = np.arange(80000, 80000 + length_of_trial, 1)

length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[8]
windows_of_non_patterned_behaviour[8] = np.arange(109000, 109000 + length_of_trial, 1)

length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[0]
windows_of_non_patterned_behaviour[0] = np.arange(356000, 356000 + length_of_trial, 1)

length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[5]
windows_of_non_patterned_behaviour[5] = np.arange(374000, 374000 + length_of_trial, 1)

#   OLD CODE TO GET THE WINDOWS
'''
#   There are more (just about) continuous stretches than patterned behaviour trials so for each length of frames of a
#   patterned behaviour trial pick the same length of frames from a different continuous stretch of non patterned
#   behaviour (making sure the non patterned stretch has enough frames in it).

#   THE FOLLOWING CODE WILL WORK ONLY IF THE FOLLOWING LINE RETURNS FALSE
print(np.any(np.sort(lengths_of_windows_of_patterned_behaviour) -
             np.sort(lengths_of_non_patterned_behaviour_frame_ranges[:number_of_patterned_events]) > 0))

windows_of_non_patterned_behaviour = []
for index_of_trial in np.arange(number_of_patterned_events):
    length_of_trial = np.sort(lengths_of_windows_of_patterned_behaviour)[index_of_trial]
    index_of_frame_range_of_non_patterned_behaviour = np.argsort(lengths_of_non_patterned_behaviour_frame_ranges)[4 + index_of_trial]
    frame_range_of_non_patterned_behaviour = long_continous_non_patterned_behaviour_frame_ranges[
        index_of_frame_range_of_non_patterned_behaviour]
    start_point_value = int(np.random.choice(frame_range_of_non_patterned_behaviour[: - length_of_trial]))
    windows_of_non_patterned_behaviour.append(np.arange(start_point_value, start_point_value + length_of_trial, 1))

lengths_of_windows_of_non_patterned_behaviour = [len(w) for w in windows_of_non_patterned_behaviour]
'''

starts_of_windows_of_non_patterned_behaviour = [s[0] for s in windows_of_non_patterned_behaviour]

windows_of_non_patterned_behaviour = [x.astype(np.int32) for _, x in sorted(zip(starts_of_windows_of_non_patterned_behaviour,
                                                               windows_of_non_patterned_behaviour),
                                                           key=lambda pair: pair[0])]

lengths_of_windows_of_non_patterned_behaviour = [len(windows_of_non_patterned_behaviour[i])
                                                 for i in np.arange(number_of_patterned_events)]

# <editor-fold desc="Have a look at the selected non patterned behaviour events">
frame = 0
ev_video = event_dataframes['ev_video']
starts_of_non_patterned_behaviour = [s[0] for s in windows_of_non_patterned_behaviour]


def tp_to_frame(frame):
    return sync_funcs.time_point_to_frame_from_video_df(ev_video, frame)[0] - frames_before_tb


dd.connect_repl_var(globals(), 'starts_of_non_patterned_behaviour', 'frame')
sv.image_sequence(globals(), 'frame', 'video_file')
# </editor-fold>
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="USING THE GENERATED WINDOWS SMOOTH THE CORRESPONDING PARTS OF THE SPEED AND THE FIRING RATES">

#   Smooth speeds and spike rates for each patterned behaviour
windows_of_patterned_behaviour_list = windows_of_patterned_behaviour[0]
for k in np.arange(1, number_of_patterned_events):
    windows_of_patterned_behaviour_list = np.concatenate((windows_of_patterned_behaviour_list,
                                                          windows_of_patterned_behaviour[k]))

speeds_patterned_behaviour_0p25 = binning.rolling_window_with_step(speeds[windows_of_patterned_behaviour_list],
                                                                    np.mean, 30, 30)

distances_rat_to_poke_all_frames = np.sqrt(
    np.power(body_positions_normalised[:, 0] - poke_position[0], 2) +
    np.power(body_positions_normalised[:, 1] - poke_position[1], 2))

distance_to_poke_patterned_behaviour_0p25 = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_patterned_behaviour_list],
                                     np.mean, 30, 30)

spike_rates_patterned_behaviour_all_frames = spike_rates[:, windows_of_patterned_behaviour_list]

spike_rates_patterned_behaviour_0p25 = \
    binning.rolling_window_with_step(spike_rates_patterned_behaviour_all_frames, np.mean, 30, 30)

#   Smooth speeds and spike rates for each non patterned trial
windows_of_non_patterned_behaviour_list = windows_of_non_patterned_behaviour[0]
for k in np.arange(1, number_of_patterned_events):
    windows_of_non_patterned_behaviour_list = np.concatenate((windows_of_non_patterned_behaviour_list,
                                                              windows_of_non_patterned_behaviour[k]))

speeds_non_patterned_behaviour_0p25 = binning.rolling_window_with_step(speeds[windows_of_non_patterned_behaviour_list],
                                                                        np.mean, 30, 30)
distance_to_poke_non_patterned_behaviour_0p25 = \
    binning.rolling_window_with_step(distances_rat_to_poke_all_frames[windows_of_non_patterned_behaviour_list],
                                                                    np.mean, 30, 30)

spike_rates_non_patterned_behaviour_all_frames = spike_rates[:, windows_of_non_patterned_behaviour_list]

spike_rates_non_patterned_behaviour_0p25 = \
    binning.rolling_window_with_step(spike_rates_non_patterned_behaviour_all_frames, np.mean, 30, 30)


#   Check that the two speed distributions are roughly the same
_ = plt.hist(speeds_non_patterned_behaviour_0p25)
_ = plt.hist(speeds_patterned_behaviour_0p25, fc=(0, 1, 0, 0.5))

#   Save all the relevant patterned and non patterned arrays (speeds, distances, time ranges and firing rates)
np.save(join(patterned_vs_non_patterned_folder, 'distances_rat_to_poke_all_frames.npy'),
        distances_rat_to_poke_all_frames)
np.save(join(patterned_vs_non_patterned_folder, 'spike_rates_patterned_behaviour_all_frames.npy'),
        spike_rates_patterned_behaviour_all_frames)
np.save(join(patterned_vs_non_patterned_folder, 'spike_rates_non_patterned_behaviour_all_frames.npy'),
        spike_rates_non_patterned_behaviour_all_frames)

np.save(join(patterned_vs_non_patterned_folder, 'windows_of_patterned_behaviour.npy'), windows_of_patterned_behaviour)
np.save(join(patterned_vs_non_patterned_folder, 'windows_of_patterned_behaviour_list.npy'),
        windows_of_patterned_behaviour_list)

np.save(join(patterned_vs_non_patterned_folder, 'windows_of_non_patterned_behaviour.npy'),
        windows_of_non_patterned_behaviour)
np.save(join(patterned_vs_non_patterned_folder, 'windows_of_non_patterned_behaviour_list.npy'),
        windows_of_non_patterned_behaviour_list)


np.save(join(patterned_vs_non_patterned_folder, 'speeds_patterned_behaviour_0p25.npy'),
        speeds_patterned_behaviour_0p25)
np.save(join(patterned_vs_non_patterned_folder, 'distance_to_poke_patterned_behaviour_0p25.npy'),
        distance_to_poke_patterned_behaviour_0p25)
np.save(join(patterned_vs_non_patterned_folder, 'spike_rates_patterned_behaviour_0p25.npy'),
        spike_rates_patterned_behaviour_0p25)

np.save(join(patterned_vs_non_patterned_folder, 'speeds_non_patterned_behaviour_0p25.npy'),
        speeds_non_patterned_behaviour_0p25)
np.save(join(patterned_vs_non_patterned_folder, 'distance_to_poke_non_patterned_behaviour_0p25.npy'),
        distance_to_poke_non_patterned_behaviour_0p25)
np.save(join(patterned_vs_non_patterned_folder, 'spike_rates_non_patterned_behaviour_0p25.npy'),
        spike_rates_non_patterned_behaviour_0p25)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="RUN THE MUTUAL INFORMATION">

#   Do the patterned behaviour vs speed
n = 0
mi_pb_spikes_vs_speed = []
for rate in spike_rates_patterned_behaviour_0p25:
    mi_pb_spikes_vs_speed.append(MI.mi_LNC([rate.tolist(), speeds_patterned_behaviour_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_pb_spikes_vs_speed = np.array(mi_pb_spikes_vs_speed)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'), mi_pb_spikes_vs_speed)

#   Do the shuffle of the best patterned behaviour vs speed
max_neuron = np.argmax(mi_pb_spikes_vs_speed)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_patterned_behaviour_0p25[max_neuron],
                                                 speeds_patterned_behaviour_0p25,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_speed_patterned_behaviour.npy'.
             format(str(max_neuron))), shuffled)


#   Do the patterned behaviour vs distance to poke
n = 0
mi_pb_spikes_vs_distance_to_poke = []
for rate in spike_rates_patterned_behaviour_0p25:
    mi_pb_spikes_vs_distance_to_poke.append(MI.mi_LNC([rate.tolist(), distance_to_poke_patterned_behaviour_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_pb_spikes_vs_distance_to_poke = np.array(mi_pb_spikes_vs_distance_to_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke_patterned_behaviour.npy'),
        mi_pb_spikes_vs_distance_to_poke)

#   Do the shuffle of the best patterned behaviour vs distance to poke
max_neuron = np.argmax(mi_pb_spikes_vs_distance_to_poke)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_patterned_behaviour_0p25[max_neuron],
                                                 distance_to_poke_patterned_behaviour_0p25,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_distance_to_poke_patterned_behaviour.npy'.
             format(str(max_neuron))), shuffled)

#   Do the non patterned behaviour vs speed
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

#   Do the non patterned behaviour vs distance to poke
n = 0
mi_non_pb_spikes_vs_distance_to_poke = []
for rate in spike_rates_non_patterned_behaviour_0p25:
    mi_non_pb_spikes_vs_distance_to_poke.append(MI.mi_LNC([rate.tolist(), distance_to_poke_non_patterned_behaviour_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_non_pb_spikes_vs_distance_to_poke = np.array(mi_non_pb_spikes_vs_distance_to_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke_non_patterned_behaviour.npy'),
        mi_non_pb_spikes_vs_distance_to_poke)


#   Do the correlation between the distance between the rat and the poke and the firing rates
distances_rat_to_poke_all_frames = np.sqrt(
    np.power(body_positions_normalised[:, 0] - poke_position[0], 2) +
    np.power(body_positions_normalised[:, 1] - poke_position[1], 2))

distances_rat_to_poke_all_frames_0p25 = binning.rolling_window_with_step(distances_rat_to_poke_all_frames,
                                                                         np.mean, 30, 30)

n = 0
mi_spikes_vs_distance_to_poke = []
for rate in spike_rates_0p25:
    mi_spikes_vs_distance_to_poke.append(MI.mi_LNC([rate.tolist(), distances_rat_to_poke_all_frames_0p25[:-1]],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spikes_vs_distance_to_poke = np.array(mi_spikes_vs_distance_to_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke.npy'),
        mi_spikes_vs_distance_to_poke)

#   Do the shuffle of the best distance to poke
max_neuron = np.argmax(mi_spikes_vs_distance_to_poke)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_0p25[max_neuron],
                                                 distances_rat_to_poke_all_frames_0p25[:-1],
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_distance_to_poke.npy'.
             format(str(max_neuron))), shuffled)

#   Do the correlation between the distance between rat and ball and the firing rates

distances_rat_to_ball_0p25 = binning.rolling_window_with_step(distances_rat_to_ball, np.mean, 30, 30)

frames_of_ball_on = ball_and_rat_positions_in_frame['frame'].values.astype(int)
spike_rates_with_ball_on = spike_rates[:, frames_of_ball_on]
spike_rates_with_ball_on_0p25 = binning.rolling_window_with_step(spike_rates_with_ball_on, np.mean, 30, 30)

n = 0
mi_spikes_vs_distance_to_ball = []
for rate in spike_rates_with_ball_on_0p25:
    mi_spikes_vs_distance_to_ball.append(MI.mi_LNC([rate.tolist(), distances_rat_to_ball_0p25],
                                           k=10, base=np.exp(1), alpha=0.4, intens=1e-10))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spikes_vs_distance_to_ball = np.array(mi_spikes_vs_distance_to_ball)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_ball.npy'),
        mi_spikes_vs_distance_to_ball)

#   Do the shuffle of the best distance to ball
max_neuron = np.argmax(mi_spikes_vs_distance_to_ball)

shuffled, mean, conf_intervals = MI.shuffle_test(MI.mi_LNC,  spike_rates_with_ball_on_0p25[max_neuron],
                                                 distances_rat_to_ball_0p25,
                                                 z=False, ns=1000, ci=0.95, k=10, base=np.exp(1), alpha=0.4,
                                                 intens=1e-10)
np.save(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_1060_vs_distance_to_ball.npy'), shuffled)


#   Do the correlation between the distance of the rat to the ball with the spike rates conditioned on the distance
#   of the rat to the poke

frames_of_ball_on = ball_and_rat_positions_in_frame['frame'].values.astype(int)
spike_rates_with_ball_on = spike_rates[:, frames_of_ball_on]
spike_rates_with_ball_on_0p25 = binning.rolling_window_with_step(spike_rates_with_ball_on, np.mean, 30, 30)
spike_rates_with_ball_on_0p25_cmi = np.expand_dims(spike_rates_with_ball_on_0p25, axis=2).tolist()

distances_rat_to_poke_all_frames = np.sqrt(
    np.power(body_positions_normalised[:, 0] - poke_position[0], 2) +
    np.power(body_positions_normalised[:, 1] - poke_position[1], 2))
distances_rat_to_poke_ball_on = distances_rat_to_poke_all_frames[frames_of_ball_on]
distances_rat_to_poke_ball_on_0p25 = binning.rolling_window_with_step(distances_rat_to_poke_ball_on,
                                                                         np.mean, 30, 30)
distances_rat_to_poke_ball_on_0p25_cmi = np.expand_dims(distances_rat_to_poke_ball_on_0p25, axis=1)


distances_rat_to_ball_0p25 = binning.rolling_window_with_step(distances_rat_to_ball, np.mean, 30, 30)
distances_rat_to_ball_0p25_cmi = np.expand_dims(distances_rat_to_ball_0p25, axis=1).tolist()


n = 0
mi_spikes_vs_distance_to_ball_conditioned_on_poke = []
for rate in spike_rates_with_ball_on_0p25:
    mi_spikes_vs_distance_to_ball_conditioned_on_poke.append(mi(spike_rates_with_ball_on_0p25_cmi[0],
                                                                distances_rat_to_ball_0p25_cmi,
                                                                z=distances_rat_to_poke_ball_on_0p25_cmi,
                                                             k=10, base=np.exp(1)))
    n += 1
    print('Done neuron {}'.format(str(n)))

mi_spikes_vs_distance_to_ball_conditioned_on_poke = np.array(mi_spikes_vs_distance_to_ball_conditioned_on_poke)
np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_ball_conditioned_on_poke.npy'),
        mi_spikes_vs_distance_to_ball_conditioned_on_poke)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS FOR PATTERNED VS NON PATTERNED">
mi_pb_spikes_vs_speed = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_speed_patterned_behaviour.npy'))
mi_non_pb_spikes_vs_speed = np.load(join(mutual_information_folder,
                                         'mutual_infos_spikes_vs_speed_non_patterned_behaviour.npy'))


# Have a look at the MIs vs the chance level
max_neuron = np.argmax(mi_pb_spikes_vs_speed)
shuffled = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_speed_patterned_behaviour.npy').
             format(str(max_neuron)))
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
    plt.vlines(x=np.cumsum(lengths_of_windows_of_patterned_behaviour) / 30, ymin=0,
               ymax=np.max(speeds_patterned_behaviour_0p25))
    return None


index = 0
out = None
sl.connect_repl_var(globals(), 'index', 'out', 'draw1', slider_limits=[0, len(speed_corr_neurons_pb_index) - 1])


neuron_indices_that_change = speed_corr_neurons_pb_index[[0, 2, 3, 4, 5, 8, 9, 12, 13, 14, 17, 18, 20, 21, 22, 24, 26,
                                                         27, 29]]


def plot_fr_of_neurons_that_change(index, fig1, fig2):
    fig1.clear()
    fig2.clear()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax1.plot(spike_rates_0p25[speed_corr_neurons_pb_index[index], :])
    ax1.vlines(x=[57318/30, 212829/30], ymin=0, ymax=spike_rates_0p25[speed_corr_neurons_pb_index[index], :].max())
    ax2.plot(speeds_patterned_behaviour_0p25)
    ax2.plot(spike_rates_patterned_behaviour_0p25[speed_corr_neurons_pb_index[index], :])


fig1 = plt.figure(1)
fig2 = plt.figure(2)
args = [fig1, fig2]
sl.connect_repl_var(globals(), 'index', 'out', 'plot_fr_of_neurons_that_change', 'args',
                    slider_limits=[0, len(neuron_indices_that_change) - 1])


def draw2(index):
    plt.clf()
    plt.plot(speeds_non_patterned_behaviour_0p25)
    plt.plot(spike_rates_non_patterned_behaviour_0p25[speed_corr_neurons_non_pb_index[index], :])
    plt.vlines(x=np.cumsum(lengths_of_windows_of_non_patterned_behaviour) / 30, ymin=0,
               ymax=np.max(speeds_non_patterned_behaviour_0p25))
    return None


index = 0
out = None
sl.connect_repl_var(globals(), 'index', 'out', 'draw2', slider_limits=[0, len(speed_corr_neurons_non_pb_index) - 1])

np.intersect1d(speed_corr_neurons_pb_index, speed_corr_neurons_non_pb_index)
print(speed_corr_neurons_non_pb_index[index])

index_of_non_pb_neurons_that_show_modulation = [334, 623, 681, 960, 1060]
frame_of_weirdness = windows_of_patterned_behaviour[14][-1]

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS FOR DISTANCE TO BALL">
mi_spikes_vs_distance_to_ball = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_ball.npy'))

shuffled_to_ball = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_distance_to_ball.npy').
             format(str(max_neuron)))
mean_sh = np.mean(shuffled_to_ball)
confidence_level = 0.99
confi_intervals = shuffled_to_ball[int((1. - confidence_level) / 2 * 1000)], shuffled_to_ball[int((1. + confidence_level) / 2 * 1000)]

plt.hist(mi_spikes_vs_distance_to_ball, bins=200, fc=(0, 1, 0, 0.5))
plt.hist(shuffled_to_ball, bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 20)

speed_corr_neurons_d_ball_index = np.squeeze(np.argwhere(mi_spikes_vs_distance_to_ball > mean_sh+10*confi_intervals[1]))
speed_corr_neurons_d_ball = template_info.loc[speed_corr_neurons_d_ball_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=speed_corr_neurons_d_ball,
                                     dot_sizes=mi_spikes_vs_distance_to_ball[speed_corr_neurons_d_ball_index] * 4000,
                                     font_size=5)

# </editor-fold>
# -------------------------------------------------


# -------------------------------------------------
# <editor-fold desc="VISUALISE RESULTS FOR DISTANCE TO POKE">
mi_spikes_vs_distance_to_poke = np.load(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_poke.npy'))


shuffled_to_poke = np.load(join(mutual_information_folder, 'shuffled_mut_info_spike_rate_{}_vs_distance_to_poke.npy').
             format(str(max_neuron)))
mean_sh = np.mean(shuffled_to_poke)
confidence_level = 0.99
confi_intervals = shuffled_to_poke[int((1. - confidence_level) / 2 * 1000)], shuffled_to_poke[int((1. + confidence_level) / 2 * 1000)]

plt.hist(mi_spikes_vs_distance_to_poke, bins=200, fc=(0, 1, 0, 0.5))
plt.hist(shuffled_to_poke, bins=200, color=(1, 0, 0, 0.4))
plt.vlines([mean_sh, mean_sh+confi_intervals[0], mean_sh+confi_intervals[1]], 0, 20)

speed_corr_neurons_d_poke_index = np.squeeze(np.argwhere(mi_spikes_vs_distance_to_poke > mean_sh+10*confi_intervals[1]))
speed_corr_neurons_d_poke = template_info.loc[speed_corr_neurons_d_poke_index]
spp.view_grouped_templates_positions(kilosort_folder, const.BRAIN_REGIONS, const.PROBE_DIMENSIONS,
                                     const.POSITION_MULT, template_info=speed_corr_neurons_d_poke,
                                     dot_sizes=mi_spikes_vs_distance_to_ball[speed_corr_neurons_d_poke_index] * 4000,
                                     font_size=5)

plt.plot(spike_rates_0p25[np.argmax(mi_spikes_vs_distance_to_poke)])
plt.plot(np.array(distances_rat_to_poke_all_frames_0p25) * 20)

# </editor-fold>
# -------------------------------------------------

mi_spikes_vs_distance_to_ball_conditioned_on_poke = \
    np.save(join(mutual_information_folder, 'mutual_infos_spikes_vs_distance_to_ball_conditioned_on_poke.npy'))

plt.hist(mi_spikes_vs_distance_to_ball_conditioned_on_poke, bins=200, fc=(0, 1, 0, 0.5))
