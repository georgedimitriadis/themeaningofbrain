
import numpy
from os import path
import importlib

csv_funcs = importlib.import_module('ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs')


CAMERA_BIT = 1
TRIAL_END_BIT = 2
BEAM_BREAK_BIT = 4
SOUND_BIT = 8


def find_nearest(array, value):
    idx = (numpy.abs(array-value)).argmin()
    return idx, array[idx]


def _generate_camera_pulses(sync_file, clean=False, cam_ttl_pulse_period=158):

    sync_data = numpy.fromfile(sync_file, numpy.uint16).astype(numpy.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    ones = sync_data_zeroed & CAMERA_BIT
    ones_up = numpy.argwhere(numpy.diff(ones) < 0).squeeze()
    camera_pulses = ones_up

    # Correcting for extra pulses appearing between camera pulses sometimes (for data set recorded with the
    # Arduino for voltage shifting plugged into the main computer)
    if clean:
        camera_pulses = list()
        camera_pulses.append(ones_up[0])
        step2_index = 0
        for index in numpy.arange(1, len(ones_up)):
            if ones_up[index] - camera_pulses[step2_index] > cam_ttl_pulse_period:
                camera_pulses.append(ones_up[index])
                step2_index += 1
        camera_pulses = numpy.array(camera_pulses).squeeze()

    return camera_pulses


def _find_beam_breaks(sync_file):
    sync_data = numpy.fromfile(sync_file, numpy.uint16).astype(numpy.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    beam_break_on = sync_data_zeroed & BEAM_BREAK_BIT
    end_beam_break = numpy.argwhere(numpy.diff(beam_break_on) > 0).squeeze()
    start_beam_break = numpy.argwhere(numpy.diff(beam_break_on) < 0).squeeze()

    return numpy.array(list(zip(start_beam_break, end_beam_break)))


def generate_events_from_sync_file(data_folder, clean=False, cam_ttl_pulse_period=158):

    sync_file = path.join(data_folder, 'Sync.bin')
    camera_pulses = _generate_camera_pulses(sync_file, clean, cam_ttl_pulse_period)

    camera_pulse_frame_number = len(camera_pulses)
    print('Number of camera TTL pulses: {}'.format(camera_pulse_frame_number))

    video_csv_file = path.join(data_folder, 'Video.csv')
    total_frames = csv_funcs.find_number_of_frames_from_video_csv(video_csv_file)
    print('Number of video frames: {}'.format(total_frames))

    print('Difference = {}'.format(camera_pulse_frame_number - total_frames))

    beam_breaks = _find_beam_breaks(sync_file)

    return camera_pulses, beam_breaks







