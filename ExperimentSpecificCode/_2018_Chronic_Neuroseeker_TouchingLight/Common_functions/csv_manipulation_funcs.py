
import csv
from os import path
import pandas as pd
import numpy as np


def find_number_of_frames_from_video_csv(video_csv_file):
    """
    Get the number of frames the camera collected (including the frames that were dropped from the Video.avi file)
    using the Video.csv file information

    :param video_csv_file: The full path to the Video.csv
    :return: The number of frames
    """
    frame_numbers = []
    with open(video_csv_file, newline='') as csvfile:
        video_csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in video_csv_reader:
            frame_numbers.append(row[1])

    return int(frame_numbers[-1]) - int(frame_numbers[0]) + 1


def video_csv_as_dataframe(video_csv_file):
    """
    Turns the Video.csv file into a data frame with columns = Date Hour Minute Second FrameAdded and FrameNumber.
    The FrameNumber column is the frame number given by the camera and takes into account any dropped frames

    :param video_csv_file: the Video.csv file
    :return: The DataFrame
    """
    video_record = pd.read_csv(video_csv_file, sep='-|T|:|\+|00 ', engine='python', header=None)
    video_record.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'FrameAdded', 'Nothing', 'FrameNumber']
    video_record = video_record.drop('Nothing', axis=1)
    return video_record


def get_true_frame_array(data_folder):
    """
    Returns an array with length the number of frames in the video.avi and each element the camera frame number that
    corresponds to this video frame.
    \n
    So if true_frame_index[5] = 8 that means that the 5th frame of the video.avi
    corresponds to the 8th frame captured by the camera (which means there had been 3 frames dropped from the video).

    :param data_folder: The folder that the Video.csv file is in
    :return: The true_frame_index
    """
    video_csv_file = path.join(data_folder, 'Video.csv')
    video_record = video_csv_as_dataframe(video_csv_file)
    true_frame_index = np.array((video_record['FrameNumber'] - video_record['FrameNumber'].iloc[0]).tolist())

    return true_frame_index


def create_events_from_events_csv(csv_folder):
    """
    Under construction

    :param csv_folder:
    :return:
    """
    events_csv_file = path.join(csv_folder, 'Events.csv')

    date_time = []
    event_name = []
    event_info = []
    with open(events_csv_file, newline='') as csvfile:
        events_csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in events_csv_reader:
            date_time.append(row[0])
            event_name.append(row[1])
            event_info.append(row[2])


def get_rat_positions_from_bonsai_image_processing(csvfile):
    """
    Under construction

    :param csvfile:
    :return:
    """
    positions_df = pd.read_csv(csvfile, delim_whitespace=True )
    positions_df.columns = ['Frame', 'Brother XY', 'Recorded XY']


