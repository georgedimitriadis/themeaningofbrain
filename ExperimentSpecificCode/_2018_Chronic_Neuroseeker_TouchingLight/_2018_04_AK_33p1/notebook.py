

from os.path import join
import numpy as np
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.events_sync_funcs as sync_funcs
import ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs as csv_funcs
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const

import sequence_viewer as sv
import transform as tr


date_folder = 8

data_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Data')
spikes_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date_folder], 'Analysis', 'Kilosort')

sync = np.fromfile(join(data_folder, 'Sync.bin'), dtype=np.uint16).astype(np.int32)
sync -= sync.min()

camera_pulses, beam_breaks, sounds = \
    sync_funcs.generate_events_from_sync_file(data_folder, clean=True,
                                              cam_ttl_pulse_period=
                                              const.CAMERA_TTL_PULSES_TIMEPOINT_PERIOD)

first_point_of_cam_in_sync = const.start_end_ephys_points_of_video_recording[date_folder][0]
last_point_of_cam_in_sync = const.start_end_ephys_points_of_video_recording[date_folder][1]
points_with_cam_in_sync = last_point_of_cam_in_sync - first_point_of_cam_in_sync
points_per_pulse = points_with_cam_in_sync / len(camera_pulses)  # 166.688


video_csv_file = join(data_folder, 'Video.csv')
true_frame_index = csv_funcs.get_true_frame_array(video_csv_file)


def get_sync_point_from_frame(x):
    return int(first_point_of_cam_in_sync - 200 + (true_frame_index[x] * points_per_pulse))


# Show the video gui
video_frame = 0
video_file = join(data_folder, 'Video.avi')
sv.video(globals(), 'video_frame', 'video_file')

# Show the sync trace gui
sync_point = first_point_of_cam_in_sync - 200
sync_range = 2000
period = 1/20000
sv.graph_range(globals(), 'sync_point', 'sync_range', 'sync', 'period')


# Connect the sync trace gui to the video gui
def frame_to_sync(x):
    return get_sync_point_from_frame(x)


tr.connect_repl_var(globals(), 'video_frame', 'frame_to_sync', 'sync_point')


# Connect the video gui to the sync trace gui
def sync_to_frame(x):
    true_frame = int((x - first_point_of_cam_in_sync + 200)/points_per_pulse)
    video_frame = np.argwhere(true_frame_index == true_frame).squeeze()
    while video_frame.size == 0:
        true_frame -= 1
        video_frame = np.argwhere(true_frame_index == true_frame).squeeze()
    return video_frame.tolist()


tr.connect_repl_var(globals(), 'sync_point', 'sync_to_frame', 'video_frame')


# Connect the video gui to a gui showing if the sound is on
sound_on = False


def is_sound_on(x):
    sync_point = get_sync_point_from_frame(x)
    result = False
    for sound_period in sounds:
        if sound_period[0] < sync_point and sound_period[1] > sync_point:
            result = True

    return result


tr.connect_repl_var(globals(), 'video_frame', 'is_sound_on', 'sound_on')

# Connect the video gui to a gui showing if the sound is on
beam_broken = False


def is_beam_broken(x):
    sync_point = get_sync_point_from_frame(x)
    result = False
    for beam_period in beam_breaks:
        if beam_period[0] > sync_point and beam_period[1] < sync_point:
            result = True

    return result


tr.connect_repl_var(globals(), 'video_frame', 'is_beam_broken', 'beam_broken')
