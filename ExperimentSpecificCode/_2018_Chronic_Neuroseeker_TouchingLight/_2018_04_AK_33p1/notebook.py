

from os.path import join
import numpy as np
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

import sequence_viewer as sv
import video_viewer as vv
import transform as tr

import pandas as pd

date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Kilosort')

sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()



# Show the compressed video gui
video_frame = 1
video_file = join(data_folder, 'Video.avi')
vv.video(globals(), 'video_frame', 'video_file')

# Show the raw images gui
sv.image_sequence(globals(), 'video_frame', 'video_file')


camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

computer_time_amp_time = \
    sync_funcs.get_computer_time_to_amp_time_points_dataframe(data_folder, clean=True,
                                                              cam_ttl_pulse_period=
                                                              const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

for event_type in sync_funcs.event_types:
    exec(r'{} = sync_funcs.get_dataframe_of_event_csv_file(data_folder, event_type)'.format(event_type))
    print('Done with the {} event'.format(event_type))



