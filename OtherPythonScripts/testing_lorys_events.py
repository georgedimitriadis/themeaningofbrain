
from os.path import join
import numpy as np
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs

import sequence_viewer as sv
import transform as tr
import drop_down as dd
import video_viewer as vv

import re
import pandas as pd


# Define folders and load some basic data
date_folder = 8

data_folder = r'Y:\swc\kampff\friday\Lorenza\2018_04_29-15_43'
events_folder = join(data_folder, 'events')

sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()

sampling_freq = 30000
cam_ttl_pulse_period = 122
reward_sound_max_duration = 3000

#  Generate the pickles of DataFrames for most of the csv event files
for event_type in sync_funcs.event_types:
    exec(r'{} = sync_funcs.get_dataframe_of_event_csv_file(data_folder, event_type, 122)'.format(event_type))
    print('Done with the {} event'.format(event_type))
# ----------------------------------

#  Load the pre generated DataFrames for the event CSVs
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)

# Create some arrays and constants relating to the events
camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=cam_ttl_pulse_period)

points_per_pulse = np.mean(np.diff(camera_pulses))

camera_frames_in_video = csv_funcs.get_true_frame_array(data_folder)
time_point_of_first_video_frame = camera_pulses[camera_frames_in_video][0]
# ----------------------------------




sound_bit_on = sync & 8
start=0
step = 5000
sv.graph_range(globals(), 'start', 'step', 'sound_bit_on')

sound_durations = np.array([s[1] - s[0] for s in sounds])
reward_sounds_mask = sound_durations < reward_sound_max_duration
reward_sounds = sounds[reward_sounds_mask]
trial_sounds_mask = sound_durations > reward_sound_max_duration
trial_sounds = sounds[trial_sounds_mask]

trial_start_events = pd.read_pickle(join(events_folder, sync_funcs.event_types[9]+".pkl"))
trial_end_events = pd.read_pickle(join(events_folder, sync_funcs.event_types[8]+".pkl"))

trial_end_events['AmpTimePoints'] - np.array(trial_sounds)[:, 1]
trial_end_events[trial_end_events['Result']=='Food']['AmpTimePoints'] - np.array(reward_sounds)[:, 1]