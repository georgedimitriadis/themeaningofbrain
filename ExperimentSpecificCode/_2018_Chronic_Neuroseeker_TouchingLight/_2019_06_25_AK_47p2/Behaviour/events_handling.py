

from os.path import join
import numpy as np
from BrainDataAnalysis import neuroseeker_specific_functions as ns_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2019_06_25_AK_47p2 import constants as const

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
