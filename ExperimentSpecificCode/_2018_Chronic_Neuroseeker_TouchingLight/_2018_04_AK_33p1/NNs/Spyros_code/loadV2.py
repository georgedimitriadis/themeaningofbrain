
from os.path import join
import numpy as np
import pandas as pd
import time

import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"
#data_folder = r'./NN_data/FiringRateDataPrep'
base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
data_folder = join(base_data_folder, 'FiringRateDataPrep')

from os.path import join
import numpy as np
import pandas as pd
import time

video_events_file = join(data_folder, 'Video.pkl')
sampling_freq = 20000
time_points_per_frame = 166

spike_info = np.load(join(data_folder, 'spike_info_after_cortex_sorting.df'), allow_pickle=True)
templates = np.unique(spike_info['template_after_sorting'])
video_events = np.load(join(data_folder, 'Video.pkl'), allow_pickle=True)
# </editor-fold>

# -------------------------------------------------
# # <editor-fold desc="Generate the basic spike info as a two column pandas df with times and neurons and save it (no need to run again, data is saved)">
# templates_dict = {y: x for x, y in dict(enumerate(templates)).items()}
# basic_spike_info = spike_info[['times', 'template_after_sorting']]
# t = [templates_dict[basic_spike_info['template_after_sorting'].iloc[i]] for i in np.arange(len(basic_spike_info))]
# basic_spike_info['template_after_sorting'] = t
# basic_spike_info = basic_spike_info.rename({'times': 'time', 'template_after_sorting': 'neuron'})
#
# pd.to_pickle(basic_spike_info, join(data_folder, 'basic_spike_info.pd'))
#
# # </editor-fold>


# -------------------------------------------------
# <editor-fold desc="2) Load spike information and create the function that returns the count of spikes in a frame
basic_spike_info = np.load(join(data_folder, 'basic_spike_info.pd'), allow_pickle=True)
basic_spike_info = basic_spike_info.values
video_times = video_events.values

spike_times_in_all_neurons = []
for template in np.arange(len(templates)):
    neuron_spike_times = basic_spike_info[np.squeeze(np.argwhere(basic_spike_info[:, 1] == template)), 0]
    spike_times_in_all_neurons.append(neuron_spike_times)


def frame_to_time_point(frame):
    return video_events['AmpTimePoints'].iloc[frame]


def get_spike_count_of_all_neurons_in_frame(frame):
    start_time_point = frame_to_time_point(frame)
    end_time_point = start_time_point + time_points_per_frame
    neurons_spike_count = np.zeros(len(templates))

    for n in np.arange(len(templates)):
        neuron_spike_times = spike_times_in_all_neurons[n]
        try:
            count = len(np.squeeze(np.where(np.logical_and(neuron_spike_times > start_time_point,
                                                           neuron_spike_times < end_time_point))))
        except TypeError:
            count = 1
        neurons_spike_count[n] = count
    return neurons_spike_count

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="3) Check if the results look ok
# import matplotlib.pyplot as plt
# frames = 1000
# start_frame = 10000
#
# m = []
# for f in np.arange(start_frame, start_frame + frames):
#     m.append(get_spike_count_of_all_neurons_in_frame(f))
# m = np.array(m)
#
# print(m.max())
# plt.imshow(m)

cap = cv2.VideoCapture(join(base_data_folder, 'SubsampledVideo', 'Video_undistrorted_150x112_120fps.mp4'))

full_matrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(len(templates), len(video_times))).T

def create_data():
    import progressbar

    bar = progressbar.ProgressBar(max_value=500000)
    i = 0
    X = []
    Y = []

    frames_per_packet = 360
    X_current_buffer = []
    Y_current_buffer = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y = cv2.resize(y, (150 // 4, 112 // 4), interpolation=cv2.INTER_AREA)
        # print(resized.shape)
        # plt.imshow(resized)
        # plt.show()
        x = get_spike_count_of_all_neurons_in_frame(i)
        X_current_buffer.append(x)
        Y_current_buffer.append(y)
        bar.update(i)
        i += 1
        if (i % frames_per_packet) == 0:
            X.append(X_current_buffer)
            Y.append(Y_current_buffer)
            X_current_buffer = []
            Y_current_buffer = []

        # if (i > 20000):
        #     break

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    np.savez("data.npz", X=X, Y=Y)



# def create_data_rolling():
#     import progressbar
#
#     bar = progressbar.ProgressBar(max_value=500000)
#     i = 0
#     X = []
#     Y = []
#
#     frames_per_packet = 360
#     X_current_buffer = []
#     Y_current_buffer = []
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         y = cv2.resize(y, (150 // 4, 112 // 4), interpolation=cv2.INTER_AREA)
#         # print(resized.shape)
#         # plt.imshow(resized)
#         # plt.show()
#         x = get_spike_count_of_all_neurons_in_frame(i)
#         X_current_buffer.append(x)
#         Y_current_buffer.append(y)
#         bar.update(i)
#         i += 1
#         if (len(X_current_buffer) == frames_per_packet):
#             X.append(X_current_buffer)
#             Y.append(Y_current_buffer)
#             X_current_buffer.pop(0)
#             Y_current_buffer.pop(0)
#
#         if (i > 20000):
#             break
#
#     X = np.array(X)
#     Y = np.array(Y)
#     print(X.shape)
#     print(Y.shape)
#     np.savez("data.npz", X=X, Y=Y)

#
# def sample_data(batch_size,):
#     import progressbar
#
#     X_0 = []
#     r = []
#     Y = []
#
#     frames_per_packet = 360
#
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     bar = progressbar.ProgressBar(max_value=batch_size)
#     for i in range(batch_size):
#         X_current_buffer = []
#         Y_current_buffer = []
#
#
#         r_int = np.random.randint(total - frames_per_packet )
#         r.append(r_int)
#
#
#
#         for j in range(frames_per_packet):
#             cap.set(1, r_int + j)
#             ret, frame = cap.read()
#             x = get_spike_count_of_all_neurons_in_frame(r_int+j)
#             X_current_buffer.append(x)
#             y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             y = cv2.resize(y, (150 // 4, 112 // 4), interpolation=cv2.INTER_AREA)
#             Y_current_buffer.append(y)
#         X_0.append(X_current_buffer)
#         Y.append(Y_current_buffer)
#         bar.update(i)
#
#     X_0 = np.array(X_0)
#     r = np.array(r)
#     Y = np.array(Y)
#
#
#     np.savez("data_random.npz", r = r,  X=X_0, Y=Y)


def sample_data(batch_size,):
    import progressbar

    X_0 = []
    r = []
    Y = []

    frames_per_packet = 360

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = progressbar.ProgressBar(max_value=batch_size)
    for i in range(batch_size):
        X_current_buffer = []
        Y_current_buffer = []

        if i < batch_size * 0.1:
            start = 0
            end = int(batch_size * 0.9)
        else:
            start = int(batch_size * 0.9)
            end = batch_size
        r_int = np.random.randint(start, end - frames_per_packet )
        r.append(r_int)

        for j in range(frames_per_packet):

            x = full_matrix[r_int + j]
            X_current_buffer.append(np.array(x, dtype= np.float32, copy = False))
            if(frames_per_packet-1 == j or j == 0):
                cap.set(1, r_int + j)
                ret, frame = cap.read()
                y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                y = cv2.resize(y, (150 // 4, 112 // 4), interpolation=cv2.INTER_AREA)
                Y_current_buffer.append(np.array(y,dtype= np.float32, copy = False))
        X_0.append(X_current_buffer)
        Y.append(Y_current_buffer)
        bar.update(i)

    X_0 = np.array(X_0, dtype= np.float32, copy = False)
    r = np.array(r, dtype= np.float32, copy = False)
    Y = np.array(Y, dtype= np.float32, copy = False)


    np.savez(join(data_folder, "data_random-fullV2.npz"), r = r,  X=X_0, Y=Y)


