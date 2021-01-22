
from os.path import join
import numpy as np
import pandas as pd
import time

import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"

# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Sparce\NNs
# or
# F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NeuropixelSimulations\Long\NNs

base_data_folder = r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs'
data_folder = join(base_data_folder, 'FiringRateDataPrep')
save_data_folder = join(base_data_folder, 'Data', 'TimeSeriesSplit', 'data_100KsamplesEvery2Frames_5secslong_halfsizeres')

video_events_file = join(data_folder, 'Video.pkl')
sampling_freq = 20000
time_points_per_frame = 166

#spike_info = np.load(join(data_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)
spike_info = np.load(join(data_folder, 'spike_info_after_cortex_sorting.df'), allow_pickle=True) #spike_info_after_cortex_sorting.df for the Full NeuroSeeker probe
templates = np.unique(spike_info['template_after_sorting'])
video_events = np.load(join(data_folder, 'Video.pkl'), allow_pickle=True)
video_times = video_events.values

num_of_neurons = len(templates)
num_of_frames = len(video_times)

del spike_info
del templates
del video_events

video_folder = join(r'F:\Neuroseeker chronic\AK_33.1\2018_04_30-11_38\Analysis\NNs', 'SubsampledVideo')

# </editor-fold>

'''
# -------------------------------------------------
# <editor-fold desc="Generate the basic spike info as a two column pandas df with times and neurons and save it (no need to run again, data is saved)">

templates_dict = {y: x for x, y in dict(enumerate(templates)).items()}
basic_spike_info = spike_info[['times', 'template_after_sorting']]
t = [templates_dict[basic_spike_info['template_after_sorting'].iloc[i]] for i in np.arange(len(basic_spike_info))]
basic_spike_info['template_after_sorting'] = t
basic_spike_info = basic_spike_info.rename({'times': 'time', 'template_after_sorting': 'neuron'})

pd.to_pickle(basic_spike_info, join(data_folder, 'basic_spike_info.pd'))

# </editor-fold>

# -------------------------------------------------
# <editor-fold desc="Load spike information and create the function that returns the count of spikes in a frame">
basic_spike_info = np.load(join(data_folder, 'basic_spike_info.pd'), allow_pickle=True)
basic_spike_info = basic_spike_info.values
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
# <editor-fold desc="Check if the results look ok">
import matplotlib.pyplot as plt
frames = 1000
start_frame = 10000

m = []
for f in np.arange(start_frame, start_frame + frames):
    m.append(get_spike_count_of_all_neurons_in_frame(f))
m = np.array(m)

print(m.max())
plt.imshow(m)
# </editor-fold>

# Create the full_matrix
file = join(data_folder, r'full_firing_matrix.npy')
full_matrix = np.memmap(file, dtype=np.int16, mode='w+', shape=(num_of_neurons, num_of_frames))

for f in np.arange(0, num_of_frames):
    full_matrix[:, f] = get_spike_count_of_all_neurons_in_frame(f)
    if f%1000==0:
        print(f)
'''

# -------------------------------------------------
# <editor-fold desc="Create dataset">
cap = cv2.VideoCapture(join(video_folder, 'Video_undistrorted_150x112_120fps.mp4'))

full_matrix = np.memmap(join(data_folder, "full_firing_matrix.npy"),
                        dtype=np.int16, mode='r', shape=(num_of_neurons, num_of_frames)).T


def sample_data(frames_per_packet, batch_size, start_frame_for_period=None, batch_step=1):
    import progressbar

    #X_0 = []
    r = []
    #Y = []

    X_buffer = np.memmap(join(save_data_folder, 'X_buffer.npy'), dtype=np.float32, mode='w+',
                         shape=(batch_size, frames_per_packet, full_matrix.shape[1]))
    Y_buffer = np.memmap(join(save_data_folder, 'Y_buffer.npy'), dtype=np.float32, mode='w+',
                         shape=(batch_size, 2, 112 // 2, 150 // 2))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = progressbar.bar.ProgressBar(max_value=batch_size)
    for i in range(batch_size):
        #X_current_buffer = []
        #Y_current_buffer = []

        if start_frame_for_period == None:
            r_int = np.random.randint(total - frames_per_packet )
        else:
            r_int = start_frame_for_period + i * batch_step
        r.append(r_int)

        for j in range(frames_per_packet):

            x = full_matrix[r_int + j]
            #X_current_buffer.append(np.array(x, dtype=np.float32, copy=False))
            X_buffer[i, j, :] = np.array(x, dtype=np.float32, copy=False)
            if j == frames_per_packet-1 or j == 0:
                if j == 0:
                    dt = 0
                    p = 0
                else:
                    dt = frames_per_packet
                    p = 1
                cap.set(1, r_int + dt)
                ret, frame = cap.read()
                y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                y = cv2.resize(y, (150 // 2, 112 // 2), interpolation=cv2.INTER_AREA)
                #Y_current_buffer.append(np.array(y, dtype=np.float32, copy=False))
                Y_buffer[i, p, :, :] = np.array(y, dtype=np.float32, copy=False)
        #X_0.append(X_current_buffer)
        #Y.append(Y_current_buffer)
        bar.update(i)

    #X_0 = np.array(X_0, dtype=np.float32, copy=False)
    r = np.array(r, dtype=np.float32, copy=False)
    #Y = np.array(Y, dtype=np.float32, copy=False)

    #np.savez(join(save_data_folder, filename_to_save), r=r,  X=X_0, Y=Y)
    #np.savez(join(r'E:\George', filename_to_save), r=r, X=X_buffer, Y=Y_buffer)
    np.savez(join(save_data_folder, 'binary_headers.npz'), dtype=[np.float32],
             shape_X=[batch_size, frames_per_packet, full_matrix.shape[1]],
             shape_Y=[batch_size, 2, 112 // 2, 150 // 2],
             r=r)

    print('/nStart frame = {}, End frame = {}'.format(r[0], r[-1]))


# For random sampling
#sample_data("data_25000randompoints_7secslong_halfsizeres.npz", 7*120, 25000,
#            start_frame_for_period=None, batch_step=1)

frames_per_packet = 5 * 120
batch_step = 2
batch_size = num_of_frames // batch_step - 2 * frames_per_packet

# Data set that will allow TimeSeriesSplit (with n=10) with a 2 frame jump and 108K samples (so that a 1/10 chunk has 10K samples in it)
sample_data(frames_per_packet, batch_size,
            start_frame_for_period=frames_per_packet, batch_step=batch_step)

# </editor-fold>
