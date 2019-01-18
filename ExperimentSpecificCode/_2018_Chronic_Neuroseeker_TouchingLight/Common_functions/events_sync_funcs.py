
import numpy
from os import path
import importlib

csv_funcs = importlib.import_module('ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs')


bits = {'CAMERA': 1, 'BEAM_BREAK': 4, 'SOUND': 8}
# There should be but there isn't a bit=2 for trial end

def find_nearest(array, value):
    idx = (numpy.abs(array-value)).argmin()
    return idx, array[idx]


def _generate_camera_pulses(sync_file, clean=False, cam_ttl_pulse_period=158):

    sync_data = numpy.fromfile(sync_file, numpy.uint16).astype(numpy.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    ones = sync_data_zeroed & bits['CAMERA']
    ones_up = numpy.argwhere(numpy.diff(ones) < 0).squeeze()
    camera_pulses = ones_up

    # Correcting for extra pulses appearing between camera pulses sometimes (for data set recorded with the
    # Arduino for voltage shifting plugged into the main computer and causing noise in the sync trace)
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


def _find_single_bit_in_sync(sync_file, bit):
    """
    Finds the position in the sync data of the requested bit
    :param sync_file:
    :param bit:
    :return:
    """
    sync_data = numpy.fromfile(sync_file, numpy.uint16).astype(numpy.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    bit_on = sync_data_zeroed & bits[bit]
    bit_start = numpy.argwhere(numpy.diff(bit_on) < 0).squeeze()

    return bit_start


def _find_pairs_of_bits_in_sync(sync_file, bit):
    """
    Finds the start and end positions in the sync data of the requested bit
    :param sync_file:
    :param bit:
    :return:
    """
    sync_data = numpy.fromfile(sync_file, numpy.uint16).astype(numpy.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    bit_on = sync_data_zeroed & bits[bit]
    bit_start = numpy.argwhere(numpy.diff(bit_on) > 0).squeeze()
    bit_end = numpy.argwhere(numpy.diff(bit_on) < 0).squeeze()

    return numpy.array(list(zip(bit_start, bit_end)))


def generate_events_from_sync_file(data_folder, clean=False, cam_ttl_pulse_period=158):

    sync_file = path.join(data_folder, 'Sync.bin')
    camera_pulses = _generate_camera_pulses(sync_file, clean, cam_ttl_pulse_period)

    camera_pulse_frame_number = len(camera_pulses)
    print('Number of camera TTL pulses: {}'.format(camera_pulse_frame_number))

    video_csv_file = path.join(data_folder, 'Video.csv')
    total_frames = csv_funcs.find_number_of_frames_from_video_csv(video_csv_file)
    print('Number of video frames: {}'.format(total_frames))

    print('Difference = {}'.format(camera_pulse_frame_number - total_frames))

    beam_breaks = _find_pairs_of_bits_in_sync(sync_file, 'BEAM_BREAK')

    sounds = _find_pairs_of_bits_in_sync(sync_file, 'SOUND')

    return camera_pulses, beam_breaks, sounds







