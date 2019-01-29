

from os.path import join
import numpy as np
import BrainDataAnalysis.neuroseeker_specific_functions as ns_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

import sequence_viewer as sv
import transform as tr
import one_shot_viewer as osv


date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Kilosort')

time_points_buffer = 1000

lfp_data = ns_funcs.load_binary_amplifier_data(join(data_folder, 'Amplifier_LFPs.bin'),
                                               number_of_channels=const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE)

lfp_data_panes = np.swapaxes(np.reshape(lfp_data, (lfp_data.shape[0], int(lfp_data.shape[1] / time_points_buffer), time_points_buffer)), 0, 1)

ap_data = ns_funcs.load_binary_amplifier_data(join(data_folder, 'Amplifier_APs.bin'),
                                              number_of_channels=const.NUMBER_OF_AP_CHANNELS_IN_BINARY_FILE)

ap_data_panes = np.swapaxes(np.reshape(ap_data, (ap_data.shape[0], int(ap_data.shape[1] / time_points_buffer), time_points_buffer)), 0, 1)


pane = 120
colormap = 'RdYlBu'
image_levels = [0, 150]
sv.image_sequence(globals(), 'pane', 'ap_data_panes', image_levels=image_levels, colormap=colormap, flip='ud')


lfp_channels_on_probe = np.arange(9, 1440, 20)
channels_heights = ns_funcs.get_channels_heights_for_spread_calulation(lfp_channels_on_probe)
bad_lfp_channels = [35, 36, 37]
lfp_channels_used = np.delete(np.arange(const.NUMBER_OF_LFP_CHANNELS_IN_BINARY_FILE), bad_lfp_channels)


def spread_lfp_pane(p):
    pane = lfp_data_panes[p, :, :]
    spread = ns_funcs.spread_data(pane, channels_heights, lfp_channels_used)
    spread = np.flipud(spread)
    return np.expand_dims(spread, 0)

pane_data = None
tr.connect_repl_var(globals(), 'pane', 'spread_lfp_pane', 'pane_data')

osv.graph(globals(), 'pane_data')




camera_pulses, beam_breaks, sounds = \
    sync_funcs.get_time_points_of_events_in_sync_file(data_folder, clean=True,
                                                      cam_ttl_pulse_period=
                                                      const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)
points_per_pulse = np.mean(np.diff(camera_pulses))

camera_frames_in_video = csv_funcs.get_true_frame_array(data_folder)
time_point_of_first_video_frame = camera_pulses[camera_frames_in_video][0]

video_frame = 0
video_file = join(data_folder, 'Video.avi')
sv.image_sequence(globals(), 'video_frame', 'video_file')


def pane_to_frame(x):
    return sync_funcs.time_point_to_frame(time_point_of_first_video_frame, camera_frames_in_video,
                                                               points_per_pulse, x * time_points_buffer)

tr.connect_repl_var(globals(), 'pane', 'pane_to_frame', 'video_frame')