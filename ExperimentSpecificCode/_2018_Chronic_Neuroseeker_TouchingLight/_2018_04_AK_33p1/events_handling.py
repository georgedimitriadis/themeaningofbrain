

from os.path import join
import numpy as np
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

import sequence_viewer as sv
import transform as tr
import drop_down as dd
import video_viewer as vv

import re
import pandas as pd


# Define folders and load some basic data
date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Kilosort')
events_folder = join(data_folder, 'events')

sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()
# ----------------------------------


#  Generate the pickles of DataFrames for most of the csv event files
'''
for event_type in sync_funcs.event_types:
    exec(r'{} = sync_funcs.get_dataframe_of_event_csv_file(data_folder, event_type)'.format(event_type))
    print('Done with the {} event'.format(event_type))
'''
# ----------------------------------


#  Load the pre generated DataFrames for the event CSVs
event_dataframes = ns_funcs.load_events_dataframes(events_folder, sync_funcs.event_types)

# Create some arrays and constants relating to the events
camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
points_per_pulse = np.mean(np.diff(camera_pulses))

camera_frames_in_video = csv_funcs.get_true_frame_array(data_folder)
time_point_of_first_video_frame = camera_pulses[camera_frames_in_video][0]
# ----------------------------------


# HAVE A LOOK

# Show the video gui
video_frame = 0
video_file = join(data_folder, 'Video.avi')
sv.image_sequence(globals(), 'video_frame', 'video_file')
# ----------------------------------

# Show the sync trace gui
sync_range = 2000
sync_point = time_point_of_first_video_frame - int(sync_range/2)
sv.graph_range(globals(), 'sync_point', 'sync_range', 'sync')
# ----------------------------------


# Connect the video gui to the sync trace gui
def time_point_to_frame(x):
    return sync_funcs.time_point_to_frame(time_point_of_first_video_frame, camera_frames_in_video,
                                          points_per_pulse, x)


tr.connect_repl_var(globals(), 'sync_point', 'time_point_to_frame', 'video_frame')
# ----------------------------------


# Connect the sync trace gui to the video gui
def frame_to_time_point(x):
    return sync_funcs.frame_to_time_point(time_point_of_first_video_frame, camera_frames_in_video,
                                                     points_per_pulse, x)


tr.connect_repl_var(globals(), 'video_frame', 'frame_to_time_point', 'sync_point')
# ----------------------------------

# Connect the video gui to a gui showing if the sound is on
sound_on = False


def is_sound_on(x):
    time_point = sync_funcs.frame_to_time_point(time_point_of_first_video_frame, camera_frames_in_video,
                                                     points_per_pulse, x)
    result = False
    for sound_period in sounds:
        if sound_period[0] < time_point and sound_period[1] > time_point:
            result = True

    return result


tr.connect_repl_var(globals(), 'video_frame', 'is_sound_on', 'sound_on')
# ----------------------------------

# Connect the video gui to a gui showing if the beam is broken
beam_broken = False


def is_beam_broken(x):
    time_point = sync_funcs.frame_to_time_point(time_point_of_first_video_frame, camera_frames_in_video,
                                                points_per_pulse, x)
    result = False
    for beam_period in beam_breaks:
        if beam_period[0] < time_point and beam_period[1] > time_point:
            result = True

    return result


tr.connect_repl_var(globals(), 'video_frame', 'is_beam_broken', 'beam_broken')
# ----------------------------------


trial_end_points = event_dataframes['ev_rat_touch_ball']['AmpTimePoints'].tolist()
dd.connect_repl_var(globals(), 'trial_end_points', 'sync_point')


key_frame = 0
vv.video(globals(), 'key_frame', 'video_file')