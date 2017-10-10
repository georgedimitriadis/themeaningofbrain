
from GUIs.Kilosort import clean_kilosort_templates as clean
from os.path import join

base_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Analysis\Kilosort'
data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_LP3p5KHz_uV.bin')

probe_info_folder = r'E:\Software\Develop\Source\Repos\Python35Projects\TheMeaningOfBrain\Layouts\Probes\Neuroseeker'
prb_file = join(probe_info_folder, 'prb.txt')

time_points = 100
sampling_frequency = 20000

number_of_channels_in_binary_file = 1440


clean.cleanup_kilosorted_data(base_folder, number_of_channels_in_binary_file, binary_data_filename, prb_file,
                              type_of_binary=np.int16, sampling_frequency=20000,num_of_shanks_for_vis=5)


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
import scipy.signal as ssig
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

number_of_channels_in_binary_file = 1440

data_folder = r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data'
binary_data_filename = join(data_folder, r'2017_05_26T13_28_10_Amp_S16_HP3p5KHz_uV.bin')
video_file = r'video_13_28_27.1311872.avi'
pulse_data_trace_filename = r'2017_05_26T13_28_10_Sync_U16.bin'

raw_data = np.memmap(binary_data_filename, dtype=np.int16, mode='r')
number_of_timepoints_in_raw = int(raw_data.shape[0] / number_of_channels_in_binary_file)
raw_data = np.reshape(raw_data, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

pulse_data = np.memmap(join(data_folder, pulse_data_trace_filename), dtype=np.uint16, mode='r')

'''
top_of_pulse_points = np.argwhere(pulse_data==65279)
bottom_of_pulse_points = np.argwhere(pulse_data==65278)

chew_timepoints = 5*60*20000 + top_of_pulse_points[0][0]
scratch_timepoint = 51*60*20000 + top_of_pulse_points[0][0]
'''

plt.interactive(True)
fig = plt.figure()
ax = fig.add_subplot(111)

sampling_frequency = 20000
start_time = 0
end_time = 4200
step_time = 20
time_window = 0.5
num_of_windows = int((end_time - start_time) / step_time)
for win in range(num_of_windows):
    st = start_time + step_time * win
    et = st + time_window
    t = np.linspace(st, et, time_window * sampling_frequency)
    square = ssig.square(2 * np.pi * 119.61485 * (t + 20/sampling_frequency), duty=0.9) / 2 + 0.5 + 65278

    stp = int(st * sampling_frequency)
    etp = int(et * sampling_frequency)

    ax.clear()
    ax.plot(t, square, t, pulse_data[stp:etp])

    plt.waitforbuttonpress()



sampling_frequency = 20000
full_time = pulse_data.shape[0]/ sampling_frequency
t = np.linspace(0, full_time, pulse_data.shape[0])
square = ssig.square(2 * np.pi * 119.61485 * (t + 20/sampling_frequency), duty=0.91) / 2 + 0.5 + 65278
square[:265326] = 65278
np.save(r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data\corrected_camera_ttl_pulses.npy', square)
t2 = np.load(r'F:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_Chronic_Rat_22.1\2017_05_26\13_28_10\Data\corrected_camera_ttl_pulses.npy')

stp = int(10 * sampling_frequency)
etp = int(14 * sampling_frequency)
plt.plot(t, square, t, pulse_data[stp:etp])


full_time = pulse_data.shape[0] / 20000
t = np.linspace(0, full_time, pulse_data.shape[0])
square = ssig.square(2 * np.pi * 119.61485 * (t + 20/sampling_frequency), duty=0.91) / 2 + 0.5 + 65278
square[:65326] = 65278








cap = cv2.VideoCapture(join(data_folder, video_file))

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1000 / frame_rate

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
    err, frame = cap.read()
    cv2.imshow("Movie", frame)


cv2.namedWindow('Movie')
cv2.createTrackbar('Frame', 'Movie', 0, length, onChange)

onChange(0)
cv2.waitKey()

start = cv2.getTrackbarPos('Frame', 'Movie')

cap.set(cv2.CAP_PROP_POS_FRAMES, start)
play_video = True
stop = False
while cap.isOpened():
    if stop:
        break

    while play_video:
        err, frame = cap.read()
        cv2.imshow('Movie', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            play_video = False
            stop = True
            break
        if key & 0xFF == ord('p'):
            play_video = False
            break

    if stop:
        break

    while not play_video:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            play_video = False
            stop = True
            break
        if key & 0xFF == ord('s'):
            play_video = True
            break
        if key & 0xFF == ord(','):
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            new_frame_time = frame_time - frame_duration
            cap.set(cv2.CAP_PROP_POS_MSEC, new_frame_time)
            err, frame = cap.read()
            cv2.imshow("Movie", frame)
        if key & 0xFF == ord('.'):
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            new_frame_time = frame_time + frame_duration
            cap.set(cv2.CAP_PROP_POS_MSEC, new_frame_time)
            err, frame = cap.read()
            cv2.imshow("Movie", frame)

cap.release()
cv2.destroyAllWindows()




folder = r'Z:\n\Neuroseeker Probe Recordings\Neuroseeker Chronic Rat 22.1\2017_06_02\13_20_21'
file = r'InBehaviour_2017-06-02T13_20_21_Sync_U16_uV.bin'
pulse_data = np.memmap(join(folder, file), dtype=np.uint16, mode='r')






brain_regions = {'Parietal Cortex': 8000, 'Hypocampus CA1': 6230, 'Hypocampus DG': 5760, 'Thalamus LPMR': 4450,
                 'Thalamus Posterior': 3500, 'Thalamus VPM': 1930, 'SubThalamic': 1050}