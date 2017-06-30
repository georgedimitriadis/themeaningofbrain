
from GUIs.Kilosort import clean_kilosort_templates as clean
from os.path import join

base_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Analysis\Kilosort'
data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_LP3p5KHz_uV.bin')

probe_info_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
probe_connected_channels_file = r'neuroseeker_connected_channels_chronic_rat_22p1.npy'

time_points = 100
sampling_frequency = 20000

number_of_channels_in_binary_file = 1440


clean.cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename,
                              probe_info_folder, probe_connected_channels_file, sampling_frequency=20000)


import matplotlib.pyplot as plt
import numpy as np
raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

lfps = np.arange(9, 1430, 20)
plt.plot(raw_data[lfps, 60000:120000].T)

# ------------------------------
# Load video and ttl pulse train
import cv2
import sys
import time
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_HP3p5KHz_uV.bin')
video_file = r'video_13_28_27.1311872.avi'
pulse_data_trace_filename = r'2017_05_26T13_28_10_Sync_U16.bin'

raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

pulse_data = np.memmap(join(data_folder, pulse_data_trace_filename), dtype=np.uint16, mode='r')

plt.plot(pulse_data[15000000:20000000])

top_of_pulse_points = np.argwhere(pulse_data==65279)
bottom_of_pulse_points = np.argwhere(pulse_data==65278)

chew_timepoints = 5*60*20000 + top_of_pulse_points[0][0]
scratch_timepoint = 51*60*20000 + top_of_pulse_points[0][0]




cap = cv2.VideoCapture(join(data_folder, video_file))

pause = False
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_POS_MSEC = 0
frame_rate = cap.get(CV_CAP_PROP_FPS)
frame_duration = 1000 / frame_rate
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('Movie', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('p'):
        pause = True
    while pause:
        key_in_pause = cv2.waitKey(10)
        if key_in_pause & 0xFF == ord('s'):
            pause = False
        if key_in_pause & 0xFF == ord(','):
            frame_time = cap.get(CV_CAP_PROP_POS_MSEC)
            new_frame_time = frame_time - 10*frame_duration
            cap.set(0, new_frame_time)
            cv2.imshow('Movie', frame)
            pause = False
            cv2.waitKey(1)
            pause = True
        if key_in_pause & 0xFF == ord('.'):
            frame_time = cap.get(CV_CAP_PROP_POS_MSEC)
            new_frame_time = frame_time + 10*frame_duration
            cap.set(0, new_frame_time)
            cv2.imshow('Movie', frame)
            pause = False
            cv2.waitKey(1)
            pause = True


cap.release()
cv2.destroyAllWindows()




folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker Chronic Rat 22.1\2017_06_02\13_20_21'
file = r'InBehaviour_2017-06-02T13_20_21_Sync_U16_uV.bin'
pulse_data = np.memmap(join(folder, file), dtype=np.uint16, mode='r')


