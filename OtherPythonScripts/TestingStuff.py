


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import IO.ephys as ephys
import os
import Layouts.Grids.grids as grids
import numpy as np


layout = grids.grid_layout_128channels_cr_cl_rr_rl()

path = r"E:\George\DataDamp\Goncalo_ECoG\ECoG\Data\JPAK_75\2014_12_18-15_25"
data_path = "amplifier.bin"
data = ephys.load_raw_data(os.path.join(path, data_path), numchannels=128, dtype=np.uint16).dataMatrix

x = 16
y = 8
fig = plt.figure(figsize=(y, x))
gs1 = gridspec.GridSpec(x, y)
gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes.

for i in range(x*y):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('on')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.plot(data[i, 60000:65000])

plt.show()







# sea is 2 D array of 0 and 1s we have to find 1's group surrounded by 0's
def dfs(sea, i, j, b, h, visited):
    surround = ((-1, -1), (0, -1), (1, -1),
                (-1, 0), (1, 0),
                (-1, 1), (0, 1), (1, 1)
                )
    if can_visit(sea, i, j, b, h, visited):
        for s in surround:
            visited[(i, j)] = 1
            dfs(sea, i + s[0], j + s[1], b, h, visited)


def can_visit(sea, i, j, b, h, visited):
    if i >= 0 and j >= 0 and i < b and j < h:
        if (i, j) not in visited and sea[i][j] == 1:
            return True


def find_island(sea):
    visited = {}
    h = len(sea)
    count = 0
    for i, row in enumerate(sea):
        b = len(row)
        for j, item in enumerate(row):
            if can_visit(sea, i, j, b, h, visited):
                count += 1
                dfs(sea, i, j, b, h, visited)
    return count


sea = [[1, 1, 0, 0, 0],
       [0, 1, 0, 0, 1],
       [1, 0, 0, 1, 1],
       [0, 0, 0, 0, 0],
       [1, 0, 1, 0, 1]
       ]

print(find_island(sea))



def create_spike_triggered_events(data_raw_spikes, threshold, inter_spike_time_distance=0.01, amp_gain=1000,
                                  sampling_freq=30000, amp_y_digitization=65536, amp_y_range=10):
    scaling_factor = amp_y_range / (amp_y_digitization * amp_gain)
    data_in_V = (data_raw_spikes - np.mean(data_raw_spikes)) * scaling_factor
    inter_spike_distance = inter_spike_time_distance * sampling_freq
    samples = np.arange(0, np.shape(data_raw_spikes)[0])
    if threshold > 0:
        spike_crossings = np.array([x for x in samples if (data_in_V[x] > threshold)])
    if threshold < 0:
        spike_crossings = np.array([x for x in samples if (data_in_V[x] < threshold)])
    diff_spikes_times = np.diff(spike_crossings)
    spike_crossings = np.array([x for i, x in enumerate(spike_crossings[:-2]) if (diff_spikes_times[i] > inter_spike_distance)])
    spike_times = np.zeros(np.shape(spike_crossings))
    for i in range(len(spike_crossings)):
        spike_times[i] = data_raw_spikes[spike_crossings[i]] + \
                        np.argmax(data_raw_spikes[spike_crossings[i]:spike_crossings[i]+(1e-3*sampling_freq)])
    return spike_times, data_in_V




from sklearn.cluster import KMeans
kmeans_est_35 = KMeans(n_clusters=35)
kmeans_est_35.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float))