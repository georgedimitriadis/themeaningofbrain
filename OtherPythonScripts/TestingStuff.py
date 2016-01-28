


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import IO.ephys as ephys
import os
import Layouts.Grids.grids as grids


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









import itertools
import six
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn import mixture
import random



X = np.transpose(t_tsne)



n_components = 20
covariance_type = 'full'
n_iter = 10000
n_init=10

# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, n_init=n_init)
gmm.fit(X)

# Fit a Dirichlet process mixture of Gaussians using five components
dpgmm = mixture.DPGMM(n_components=5, covariance_type=covariance_type, n_iter=n_iter)
dpgmm.fit(X)

colors_ = list(six.iteritems(colors.cnames))
colours = []
for i in range(n_components):
    colours.append(random.choice(colors_))




color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'])


for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar) in enumerate(zip(
            clf.means_, clf._get_covars())):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=colours[i][0])

        # Plot an ellipse to show the Gaussian component
        #angle = np.arctan(u[1] / u[0])
        #angle = 180 * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()




from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

X = np.transpose(t_tsne)

db = DBSCAN(eps=0.53, min_samples=800).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, sample_size=5000))

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.plot(X[juxta_cluster_indices, 0], X[juxta_cluster_indices, 1], '*', markerfacecolor='b')

plt.title('Estimated number of clusters: %d' % n_clusters_)
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