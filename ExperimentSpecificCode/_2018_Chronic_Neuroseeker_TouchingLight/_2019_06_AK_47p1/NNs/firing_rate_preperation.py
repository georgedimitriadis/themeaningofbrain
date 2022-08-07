
from os.path import join
import numpy as np
import pandas as pd
import time

# -------------------------------------------------
# <editor-fold desc="1) Basic folder loading"
#data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NNs\FiringRateDataPrep'
#data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NeuropixelSimulations\Long\NNs\FiringRateDataPrep'
data_folder = r'D:\Neuroseeker chronic\AK_47.1\2019_07_04-11_51\Analysis\NeuropixelSimulations\Sparce\NNs\FiringRateDataPrep'

video_events_file = join(data_folder, 'Video.pkl')
sampling_freq = 20000
time_points_per_frame = 166

spike_info = np.load(join(data_folder, 'spike_info_after_cleaning.df'), allow_pickle=True)

templates = np.unique(spike_info['template_after_sorting'])
video_events = np.load(join(data_folder, 'Video.pkl'), allow_pickle=True)
# </editor-fold>

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
import matplotlib.pyplot as plt
frames = 3000
start_frame = 10000

m = []
for f in np.arange(start_frame, start_frame + frames):
    m.append(get_spike_count_of_all_neurons_in_frame(f))
m = np.array(m)

print(m.max())
plt.imshow(m)

# </editor-fold>


# -------------------------------------------------
# <editor-fold desc="4) Make a matrix with the counts for all neurons X all frames

#   Load the memmaped file
file = join(data_folder, 'full_matrix.npy')
full_matrix = np.memmap(file, dtype=np.int16, mode='r', shape=(len(templates), len(video_times)))



#   Make the matrix (DO NOT RUN AGAIN)
full_matrix = np.memmap(file, dtype=np.int16, mode='w+', shape=(len(templates), len(video_times)))

for f in np.arange(0, len(video_times)):
    full_matrix[:, f] = get_spike_count_of_all_neurons_in_frame(f)
    if f%1000==0:
        print('{} out of {}'.format(f, len(video_times)))


# </editor-fold>