
import csv
from os import path
import pandas as pd


def find_number_of_frames_from_video_csv(video_csv_file):

    frame_numbers = []
    with open(video_csv_file, newline='') as csvfile:
        video_csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in video_csv_reader:
            frame_numbers.append(row[1])

    return int(frame_numbers[-1]) - int(frame_numbers[0]) + 1


def create_events_from_events_csv(csv_folder):
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

    positions_df = pd.read_csv(csvfile, delim_whitespace=True )
    positions_df.columns = ['Frame', 'Brother XY', 'Recorded XY']




#pandas.concat([pandas.DataFrame([row], columns=['date_time', 'name', 'info']) for row in events_csv_reader],
#              ignore_index=True)