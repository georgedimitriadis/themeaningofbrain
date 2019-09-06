

from os.path import join
import numpy as np

import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_AK_47p2 import constants as const
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions \
    import events_sync_funcs as sync_funcs


import sequence_viewer as sv
import transform as tr

from BrainDataAnalysis import binning

import pandas as pd
import matplotlib.pyplot as plt

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

regressions_folder = join(results_folder, 'Regressions', 'DistanceTraveledBetweenPokes')

dlc_folder = join(analysis_folder, 'Deeplabcut')
dlc_project_folder = join(dlc_folder, 'projects', 'V1--2019-06-30')
video_file = join(dlc_folder, 'BonsaiCroping', 'Full_video.avi')

patterned_vs_non_patterned_folder = join(analysis_folder, 'Behaviour', 'PatternedVsNonPatterned')

# Load data
spike_rates_per_video_frame_filename = join(kilosort_folder, 'firing_rate_with_video_frame_window.npy')
spike_rates_per_250ms_filename = join(kilosort_folder, 'firing_rate_with_0p25s_window.npy')

event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)
ev_video = event_dataframes['ev_video']

template_info = pd.read_pickle(join(kilosort_folder, 'template_info.df'))
spike_info = pd.read_pickle(join(kilosort_folder, 'spike_info_after_cleaning.df'))

spike_rates = np.load(spike_rates_per_video_frame_filename)
spike_rates_0p25 = np.load(spike_rates_per_250ms_filename)

speeds = np.load(join(dlc_project_folder, 'post_processing', 'speeds.npy'))


distances_rat_to_poke_all_frames = np.load(join(patterned_vs_non_patterned_folder,
                                                'distances_rat_to_poke_all_frames.npy'))

distances_rat_head_to_poke_all_frames = np.load(join(patterned_vs_non_patterned_folder,
                                                'distances_head_to_poke_all_frames.npy'))

body_positions = np.load(join(dlc_project_folder, 'post_processing', 'body_positions.npy'))
head_positions = np.load(join(dlc_project_folder, 'post_processing', 'head_positions.npy'))

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="COMMON VARIABLES">
poke_position = [0.9, 0]

#   Smoothing to find the periods and do the MIs
smooth_window = 0.1
smooth_frames = int(smooth_window * 123)

poking_threshold = 0.149


distances_rat_head_to_poke_smoothed = np.array(binning.rolling_window_with_step(distances_rat_head_to_poke_all_frames,
                                                                       np.mean, smooth_frames, 1))

np.save(join(regressions_folder, 'distances_rat_head_to_poke_smoothed.npy'), distances_rat_head_to_poke_smoothed)
# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="FIND THE DISTANCE BETWEEN THE HEAD AND THE POKE (RUN ONCE)">

head_positions_normalised = 2 * head_positions / 640 - 1
distances_rat_head_to_poke_all_frames = np.sqrt(
    np.power(head_positions_normalised[:, 0] - poke_position[0], 2) +
    np.power(head_positions_normalised[:, 1] - poke_position[1], 2))

for i in np.arange(len(distances_rat_head_to_poke_all_frames)):
    if np.isnan(distances_rat_head_to_poke_all_frames[i]):
        distances_rat_head_to_poke_all_frames[i] = distances_rat_head_to_poke_all_frames[i-1]
np.save(join(patterned_vs_non_patterned_folder, 'distances_head_to_poke_all_frames.npy'),
        distances_rat_head_to_poke_all_frames)


# <editor-fold desc="Have a look at the video to see what distance of head from poke should be set as close to poke">
frame = 130
ev_video = event_dataframes['ev_video']


def show_surround_distance(frame, ax, thres):
    ax.clear()
    ax.plot(distances_rat_head_to_poke_smoothed[frame-120:frame+120])
    ax.hlines(y=thres, xmin=0, xmax= 240)
    ax.vlines(x=120, ymin=0, ymax=2)
    return None


out = None
fig = plt.figure(0)
ax = fig.add_subplot(111)

args = [ax, poking_threshold ]

sv.image_sequence(globals(), 'frame', 'video_file')
tr.connect_repl_var(globals(), 'frame', 'out', 'show_surround_distance', 'args')
# </editor-fold>

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="FIND THE FRAMES OF THE POKE VISITS">

distances_rat_head_to_poke_smoothed = np.concatenate((distances_rat_head_to_poke_all_frames[:smooth_frames],
                                                     distances_rat_head_to_poke_smoothed))

frames_away_from_poke = np.squeeze(np.argwhere(distances_rat_head_to_poke_smoothed > poking_threshold))

frames_where_rat_comes_back_to_poke = frames_away_from_poke[np.squeeze(np.argwhere(np.diff(frames_away_from_poke) > 10))]

frames_where_rat_leaves_poke = frames_away_from_poke[np.squeeze(np.argwhere(np.diff(frames_away_from_poke) > 10)) + 1]

frames_in_and_out_of_poke = np.array(list(zip(frames_where_rat_comes_back_to_poke, frames_where_rat_leaves_poke)))


frame_regions_away_from_poke_raw = np.array(list(zip(frames_where_rat_leaves_poke[:-1],
                                                     frames_where_rat_comes_back_to_poke[1:])))

frame_regions_away_from_poke = []
for fr in np.arange(len(frame_regions_away_from_poke_raw)):
    if np.diff(frame_regions_away_from_poke_raw[fr]) > 120 \
     and frame_regions_away_from_poke_raw[fr, 0] - frame_regions_away_from_poke_raw[fr - 1, 1] > 30:
        frame_regions_away_from_poke.append(frame_regions_away_from_poke_raw[fr])
frame_regions_away_from_poke = np.array(frame_regions_away_from_poke)

#   Manual corrections
frame_regions_away_from_poke[123, 1] = frame_regions_away_from_poke[124, 1]
frame_regions_away_from_poke = np.delete(frame_regions_away_from_poke, 124, 0)
frame_regions_away_from_poke[139, 1] = 493000
frame_regions_away_from_poke = np.delete(frame_regions_away_from_poke, np.arange(140, 144), 0)
frame_regions_away_from_poke[140, 1] = 500437
frame_regions_away_from_poke = np.concatenate((frame_regions_away_from_poke,
                                               [[507967, len(distances_rat_head_to_poke_smoothed) -1]]))
frame_regions_away_from_poke = np.insert(frame_regions_away_from_poke, 21, [[38080, 40019]], 0)
frame_regions_away_from_poke = np.insert(frame_regions_away_from_poke, 29, [[51493, 52680]], 0)
frame_regions_away_from_poke = np.insert(frame_regions_away_from_poke, 113, [[348422, 350503]], 0)
frame_regions_away_from_poke = np.insert(frame_regions_away_from_poke, 130, [[409185, 410315]], 0)
frame_regions_away_from_poke = np.insert(frame_regions_away_from_poke, 0, [[0, 1910]], 0)

np.save(join(regressions_folder, 'frame_regions_away_from_poke.npy'), frame_regions_away_from_poke)
'''
plt.plot(distances_rat_head_to_poke_smoothed)
plt.vlines(x=frame_regions_away_from_poke[:, 0], ymin=0, ymax=2, colors='g')
plt.vlines(x=frame_regions_away_from_poke[:, 1], ymin=0, ymax=2, colors='r')
'''

# </editor-fold>
# -------------------------------------------------

# -------------------------------------------------
# <editor-fold desc="FIND THE DISTANCE TRAVELED BETWEEN TWO POKE VISITS">

distance_traveled_from_poke = []
distance_to_travel_to_poke = np.empty(0)
distance_traveled_from_poke_in_periods = []
distance_to_travel_to_poke_in_periods = []

for period_frames in frame_regions_away_from_poke:
    distance_traveled_in_period = [0]
    distance_traveled_from_poke.append(distance_traveled_in_period[-1])
    for frame in np.arange(period_frames[0]+1, period_frames[1]):
        d = np.sqrt(np.power(body_positions[frame - 1, 0] - body_positions[frame, 0], 2) +
                    np.power(body_positions[frame - 1, 1] - body_positions[frame, 1], 2))

        if d < 1:
            d = 0

        distance_traveled_in_period.append(distance_traveled_in_period[-1] + d)
        distance_traveled_from_poke.append(distance_traveled_in_period[-1])

    distance_traveled_from_poke_in_periods.append(distance_traveled_in_period)
    distance_to_travel_to_poke_in_periods.append(distance_traveled_in_period[-1] - np.array(distance_traveled_in_period))

for t in distance_to_travel_to_poke_in_periods:
    distance_to_travel_to_poke = np.concatenate((distance_to_travel_to_poke, t))

np.save(join(regressions_folder, 'distance_traveled_from_poke.npy'), distance_traveled_from_poke)
np.save(join(regressions_folder, 'distance_to_travel_to_poke.npy'), distance_to_travel_to_poke)
np.save(join(regressions_folder, 'distance_traveled_from_poke_in_periods.npy'), distance_traveled_from_poke_in_periods)
np.save(join(regressions_folder, 'distance_to_travel_to_poke_in_periods.npy'), distance_to_travel_to_poke_in_periods)

# </editor-fold>
# -------------------------------------------------

