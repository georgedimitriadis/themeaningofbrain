
import numpy as np
from os import path
import importlib
import pandas as pd

csv_funcs = importlib.import_module('ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions.csv_manipulation_funcs')

event_types = ['BallOn',
               'BallPosition',
               'BallTracking',
               'FoodBeamBreak',
               'FoodStart',
               'RatTouchBall',
               'Tracking',
               'TrialBallType',
               'TrialEnd',
               'TrialStart',
               'Video']

csv_seperator = {'BallOn': 'T|:|\+01:00| ',
                 'BallPosition': 'T|:|\+01:00 |,|\(|\)',
                 'BallTracking': 'T|:|\+01:00 |,|\(|\)',
                 'FoodBeamBreak': 'T|:|\+01:00| ',
                 'FoodStart': 'T|:|\+01:00| ',
                 'RatTouchBall': 'T|:|\+01:00| ',
                 'Tracking': ' ',
                 'TrialBallType': 'T|:|\+01:00| ',
                 'TrialEnd': 'T|:|\+01:00| ',
                 'TrialStart': 'T|:|\+01:00| ',
                 'Video': 'T|:|\+01:00| '
                 }

csv_columns = {'BallOn': ['Date', 'Hour', 'Minute', 'Second'],
               'BallPosition': ['Date', 'Hour', 'Minute', 'Second', 'X', 'Y'],
               'BallTracking': ['Date', 'Hour', 'Minute', 'Second', 'X', 'Y'],
               'FoodBeamBreak': ['Date', 'Hour', 'Minute', 'Second'],
               'FoodStart': ['Date', 'Hour', 'Minute', 'Second'],
               'RatTouchBall': ['Date', 'Hour', 'Minute', 'Second'],
               'Tracking': ['X', 'Y'],
               'TrialBallType': ['Date', 'Hour', 'Minute', 'Second', 'Type'],
               'TrialEnd': ['Date', 'Hour', 'Minute', 'Second', 'Result'],
               'TrialStart': ['Date', 'Hour', 'Minute', 'Second'],
               'Video': ['Date', 'Hour', 'Minute', 'Second', 'CameraFrameNumber']
               }

bits = {'CAMERA': 1, 'BEAM_BREAK': 4, 'SOUND': 8}
# There should be but there isn't a bit=2 for trial end


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


def _find_single_bit_in_sync(sync_file, bit):
    """
    Finds the position in the sync data of the requested bit.
    :param sync_file: The binary file with the sync data
    :param bit: The bit number (see the bits dictionary)
    :return: The indices in the sync file where the bit appears
    """
    sync_data = np.fromfile(sync_file, np.uint16).astype(np.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    bit_on = sync_data_zeroed & bits[bit]
    bit_start = np.argwhere(np.diff(bit_on) < 0).squeeze()

    return bit_start


def _find_pairs_of_bits_in_sync(sync_file, bit):
    """
    Finds the start and end positions in the sync data of the requested bit.
    :param sync_file: The binary file with the sync data
    :param bit: The bit number (see the bits dictionary)
    :return: The array of tuples (pairs) of indices in the sync file where the bit appears and disappears
    """
    sync_data = np.fromfile(sync_file, np.uint16).astype(np.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    bit_on = sync_data_zeroed & bits[bit]
    bit_start = np.argwhere(np.diff(bit_on) > 0).squeeze()
    bit_end = np.argwhere(np.diff(bit_on) < 0).squeeze()

    return np.array(list(zip(bit_start, bit_end)))


def _find_bit_in_sync(sync_file, bit, direction):
    """
    Finds the bit in the sync file and returns the time points that there is a transition (given by the direction parameter).
    :param sync_file: The sync file to look into
    :param bit: The bit to isolate
    :param direction: 'up' or 'down' or ['up', down'] or ['down', 'up']. If 'up' the time points where the bit appears
    are returned. If 'down' the time points where the bit disappears are returned. If ['up', 'down'] then a list of
    tuples is returned where each tuple is a pair of time points where the bit first appears and then disappears. If
    ['down', 'up'] then the tuple is the time point of the bit disappearing followed by the next time point that it
    appears. The code assumes that the appearance and disappearance of the bit happens in pairs. If not things will break.

    :return: A list of integers or a list of tuples of pairs of integers being time points that the bit appears or
    disappears.
    """
    sync_data = np.fromfile(sync_file, np.uint16).astype(np.int32)
    sync_data_zeroed = sync_data - sync_data.min()
    bit_on = sync_data_zeroed & bits[bit]
    bit_start = np.argwhere(np.diff(bit_on) > 0).squeeze()
    bit_end = np.argwhere(np.diff(bit_on) < 0).squeeze()

    if direction == 'up':
        return bit_start
    elif direction == 'down':
        return bit_end
    elif direction == ['up', 'down']:
        return np.array(list(zip(bit_start, bit_end)))
    elif direction == ['down', 'up']:
        return np.array(list(zip(bit_end, bit_start)))

    return None


def _generate_camera_pulses(sync_file, clean=False, cam_ttl_pulse_period=158):
    """
    Gets the time points of the beginning of each camera pulse (and corrects for any extra noise pulses in certain
    sync files).

    :param sync_file: The binary file with the sync data
    :param clean: To clean or not the sync file
    :param cam_ttl_pulse_period: The number of points between camera pulses expected (used if clean=True)
    :return: The array of indices in the sync file (time points) where the camera sent a pulse in
    """
    bit = 'CAMERA'
    #ones_up = _find_single_bit_in_sync(sync_file, bit)
    ones_up = _find_bit_in_sync(sync_file, bit, 'up')
    camera_pulses = ones_up

    # Correcting for extra pulses appearing between camera pulses sometimes (for data set recorded with the
    # Arduino for voltage shifting plugged into the main computer and causing noise in the sync trace)
    if clean:
        camera_pulses = list()
        camera_pulses.append(ones_up[0])
        step2_index = 0
        for index in np.arange(1, len(ones_up)):
            if ones_up[index] - camera_pulses[step2_index] > cam_ttl_pulse_period:
                camera_pulses.append(ones_up[index])
                step2_index += 1
        camera_pulses = np.array(camera_pulses).squeeze()

    return camera_pulses


def _find_trial_sounds_regions_in_sync(sync_file):
    """
    Returns the pairs of time points where the trial sound was on and off.
    :param sync_file: The binary file with the sync data
    :return: The array of the tuples (pairs) of starting and ending of trial sound time points
    """
    bit = 'SOUND'
    return _find_bit_in_sync(sync_file, bit, ['up', 'down'])


def _find_trial_beam_breaks_regions_in_sync(sync_file):
    """
    Returns the pairs of time points where the trial sound was on and off.
    :param sync_file: The binary file with the sync data
    :return: The array of the tuples (pairs) of starting and ending of trial sound time points
    """
    bit = 'BEAM_BREAK'
    return _find_bit_in_sync(sync_file, bit, ['down', 'up'])


def get_time_points_of_events_in_sync_file(data_folder, clean=False, cam_ttl_pulse_period=158):
    """
    Returns all events (camera frames, beam breaks on and off and trial sounds on and off) recorded in the sync file.
    :param data_folder: The data folder where the sync file and the Video.csv are
    :param clean: To clean or not the camera pulse train of the sync file
    :param cam_ttl_pulse_period: The number of points between camera pulses expected (used if clean=True)
    :return: camera_pulses, beam_breaks, sounds
    """
    sync_file = path.join(data_folder, 'Sync.bin')
    camera_pulses = _generate_camera_pulses(sync_file, clean, cam_ttl_pulse_period)

    camera_pulse_frame_number = len(camera_pulses)
    print('Number of camera TTL pulses: {}'.format(camera_pulse_frame_number))

    video_csv_file = path.join(data_folder, 'Video.csv')
    total_frames = csv_funcs.find_number_of_frames_from_video_csv(video_csv_file)
    print('Number of video frames: {}'.format(total_frames))

    print('Difference = {}'.format(camera_pulse_frame_number - total_frames))

    beam_breaks = _find_trial_beam_breaks_regions_in_sync(sync_file)

    sounds = _find_trial_sounds_regions_in_sync(sync_file)

    return camera_pulses, beam_breaks, sounds


def get_computer_time_to_amp_time_points_dataframe(data_folder, clean=False, cam_ttl_pulse_period=158):
    """
    Returns a pandas dataframe with two columns. The first ('ComputerTime') is the Timestamps of the video frames (in
    the Video.avi file) as recorded using the computer clock. The second ('AmpTimePoints') is the time point of the
    amplifier corresponding to the beginning of this video frame.

    :param data_folder: The data folder where the sync file and the Video.csv are
    :param clean: To clean or not the camera pulse train of the sync file
    :param cam_ttl_pulse_period: The number of points between camera pulses expected (used if clean=True)
    :return: A pandas dataframe with the corresponding computer times and amplifier time points for each recorded frame
    """
    sync_file = path.join(data_folder, 'Sync.bin')
    camera_pulses = _generate_camera_pulses(sync_file, clean, cam_ttl_pulse_period)

    video_csv_file = path.join(data_folder, 'Video.csv')
    video_df = csv_funcs.video_csv_as_dataframe(video_csv_file)

    frames_zero_based = video_df['FrameNumber'] - video_df['FrameNumber'].iloc[0]
    camera_pulses_in_video = camera_pulses[frames_zero_based]

    computer_time_amp_time = pd.to_datetime(video_df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    computer_time_amp_time = pd.DataFrame(computer_time_amp_time)
    computer_time_amp_time.columns = ['ComputerTime']
    computer_time_amp_time = computer_time_amp_time.assign(AmpTimePoints=camera_pulses_in_video)

    return computer_time_amp_time


def get_amp_time_point_from_computer_time(correspondance_dataframe, datetime):
    """
    Returns the amplifier time point that is closest to a the datetime Timestamp provided.

    :param correspondance_dataframe: The dataframe created from the get_computer_time_to_amp_time_points_dataframe()
    function

    :param datetime: The datetime Timestamp to get the time points of
    :return: The time point corresponding to the datetime
    """
    items = correspondance_dataframe['ComputerTime']
    index = (np.abs(items - datetime)).values.argmin()
    t1 = items.iloc[index]
    p1 = correspondance_dataframe['AmpTimePoints'].iloc[index]
    if t1 < datetime:
        index2 = index + 1
    else:
        index2 = index - 1

    if datetime < correspondance_dataframe['ComputerTime'].iloc[0]:
        print('The datetime provided {} is smaller than the first video frame time'.format(datetime))
        dt = (correspondance_dataframe['ComputerTime'].iloc[0] - datetime)/np.timedelta64(1, 'ms')
        dp = dt * 20
        p3 = p1 - dp

    elif datetime > correspondance_dataframe['ComputerTime'].iloc[-1]:
        print('The datetime provided {} is larger than the last video frame time'.format(datetime))
        dt = (datetime - correspondance_dataframe['ComputerTime'].iloc[-1]) / np.timedelta64(1, 'ms')
        dp = dt * 20
        p3 = p1 + dp

    else:
        t2 = items.iloc[index2]
        p2 = correspondance_dataframe['AmpTimePoints'].iloc[index2]
        place = (t2 - datetime) / (t2 - t1)
        p3 = int(p2 - (p2 - p1) * place)

    return p3


def get_dataframe_of_event_csv_file(data_folder, event_type):

    if event_type is 'Video':
        csv_file_name = path.join(data_folder, event_type + '.csv')
    else:
        csv_file_name = path.join(data_folder, 'events', event_type + '.csv')

    df = pd.read_csv(csv_file_name, sep=csv_seperator[event_type], engine='python', header=None).\
        dropna(axis=1, how='all')
    df.columns = csv_columns[event_type]

    if event_type is not 'Tracking':
        days = df['Date'].str.split('-', n=2, expand=True)
        df.insert(0, 'Year', days[0])
        df.insert(1, 'Month', days[1])
        df.insert(2, 'Day', days[2])
        df.drop(columns='Date', inplace=True)

        df.insert(0, 'ComputerTime', pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]))
        df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], inplace=True)

        computer_time_amp_time = get_computer_time_to_amp_time_points_dataframe(data_folder,
                                                                                clean=True,
                                                                                cam_ttl_pulse_period=158)

        if event_type is not 'BallTracking':
            time_points = []
            for datetime in df['ComputerTime']:
                time_points.append(get_amp_time_point_from_computer_time(computer_time_amp_time, datetime))
                if len(time_points) % 1000 == 0:
                    print('Done {} points of {} points of file {}'.format(str(len(time_points)),
                                                                          str(len(df)),
                                                                          event_type))
            df.insert(1, 'AmpTimePoints', time_points)

    df.to_pickle(path.join(data_folder, 'events', event_type + '.pkl'))

    return df
