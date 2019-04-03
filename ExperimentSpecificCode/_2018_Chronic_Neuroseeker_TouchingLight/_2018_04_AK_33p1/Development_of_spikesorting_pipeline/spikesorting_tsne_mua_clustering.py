
"""
This is an effort to write code that automatically scans a t-sne embedding and finds groups of spikes within larger
blobs that come from single units.

At the end of the day it didn't work and the best thing to do is to manually try and split spikes on the t-sne
embedding
"""


from BrainDataAnalysis import tsne_analysis_functions as tsne_funcs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight._2018_04_AK_33p1 import constants as const
from spikesorting_tsne import preprocessing_kilosort_results as preproc_kilo, constants as ct, visualization as vis
from sklearn.decomposition import PCA
from spikesorting_tsne_guis import helper_functions as hf


# FOLDERS NAMES --------------------------------------------------
date = 8
kilosort_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                       'Analysis', 'Kilosort')
binary_data_filename = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                            'Data', 'Amplifier_APs.bin')
tsne_folder = join(const.base_save_folder, const.rat_folder, const.date_folders[date],
                   'Analysis', 'Tsne')
barnes_hut_exe_dir=r'E:\Software\Develop\Source\Repos\spikesorting_tsne_bhpart\Barnes_Hut\win\x64\Release'
# ----------------------------------------------------------------

'''
# ----------------------------------------------------------------
# CREATE INDIVIDUAL T-SNE FOR EACH MUA TEMPLATE WITH MORE THAN number_of_spikes_in_large_mua_templates SPIKES

preproc_kilo.t_sne_each_one_of_the_large_mua_templates_by_itself(kilosort_folder, tsne_folder, barnes_hut_exe_dir,
                                                               number_of_spikes_in_large_mua_templates=10000,
                                                               num_dims=2, perplexity=100, theta=0.2, iterations=2000,
                                                               random_seed=1, verbose=2)
# ----------------------------------------------------------------
'''

# ----------------------------------------------------------------
# HAVE A LOOK
# Find the mua templates with large spike count
number_of_spikes_in_large_mua_templates = 10000
large_mua_templates = preproc_kilo.find_large_mua_templates(kilosort_folder, number_of_spikes_in_large_mua_templates)


mua_template_to_look = 2
mua_template = large_mua_templates[mua_template_to_look]

#tsne_folder_buggy = join(tsne_folder, 'Single_MUA_templates_Bug')
spike_info = np.load(join(tsne_folder, 'template_{}'.format(mua_template), 'spike_info_original.df'))
spike_info = preproc_kilo.load_spike_info_of_template(tsne_folder, mua_template)

vis.plot_tsne_of_spikes(spike_info)


def find_spikes_in_rect(spike_info, left, right, top, bottom):
    xs = np.array(spike_info['tsne_x'].tolist())
    ys = np.array(spike_info['tsne_y'].tolist())
    index_l = np.squeeze(np.argwhere(xs > left))
    index_r = np.squeeze(np.argwhere(xs < right))
    index_t = np.squeeze(np.argwhere(ys < top))
    index_b = np.squeeze(np.argwhere(ys > bottom))
    result = np.intersect1d(np.intersect1d(index_l, index_r), np.intersect1d(index_t, index_b))
    return result


def is_single_unit(times):
    diffs, norm = hf.crosscorrelate_spike_trains(times, times, lag=1000)
    hist, edges = np.histogram(diffs, bins=50)
    if hist[25] <= 0:
        return True
    return False


def show_autocor(times):
    diffs, norm = hf.crosscorrelate_spike_trains(times, times, lag=1000)
    plt.hist(diffs, bins=50)



class Box:
    def __init__(self, spike_info, left, bottom, size, index):
        self.left = left
        self.bottom = bottom
        self.size = size

        self.spike_info = spike_info

        self.has_spikes = None
        self.is_single_unit = False

        self.index = index

        self.spikes = self.find_spikes_in_box()
        self.amount = len(self.spikes)

        self.times = None

        if self.amount > 0:
            self.times = self.find_times_in_box()
            self.has_spikes = True
            self.is_single_unit = self.check_single_unit()
        else:
            self.has_spikes = False

    def find_spikes_in_box(self):
        xs = np.array(self.spike_info['tsne_x'].tolist())
        ys = np.array(self.spike_info['tsne_y'].tolist())

        right = self.left + self.size
        top = self.bottom + self.size

        index_l = np.squeeze(np.argwhere(xs > self.left))
        index_r = np.squeeze(np.argwhere(xs < right))
        index_t = np.squeeze(np.argwhere(ys < top))
        index_b = np.squeeze(np.argwhere(ys > self.bottom))
        result = np.intersect1d(np.intersect1d(index_l, index_r), np.intersect1d(index_t, index_b))
        return result

    def find_times_in_box(self):
        return np.array(self.spike_info['times'].iloc[self.spikes].tolist())

    def check_single_unit(self):
        if len(self.times) > 4000:
            times = np.random.choice(self.times, 4000)
        else:
            times = self.times
        diffs, norm = hf.crosscorrelate_spike_trains(times, times, lag=1000)
        hist, edges = np.histogram(diffs, bins=50)
        if hist[25] <= 0:
            return True
        return False

    def show_autocor(self):
        diffs, norm = hf.crosscorrelate_spike_trains(self.times, self.times, lag=1000)
        plt.hist(diffs, bins=50)


class GroupOfBoxes():
    def __init__(self, starting_boxes=None):
        self.boxes = {}
        self.all_times = np.empty(0)

        if starting_boxes is not None:
            try:
                iterator = iter(starting_boxes)
            except TypeError:
                self.add_box(starting_boxes)
            else:
                for box in starting_boxes:
                    self.add_box(box)

    def add_box(self, box):
            self.boxes[(box.left, box.bottom)] = box
            self.all_times = np.concatenate((self.all_times, box.times))

    def check_single_unit_with_extra_box(self, box):
        if len(self.all_times) > 4000:
            times = np.random.choice(self.all_times, 4000)
        else:
            times = self.all_times

        if box.has_spikes:
            if len(box.times) > 4000:
                box_times = np.random.choice(box.times, 4000)
            else:
                box_times = box.times

            times = np.concatenate((times, box_times))
            diffs, norm = hf.crosscorrelate_spike_trains(times, times, lag=1000)
            hist, edges = np.histogram(diffs, bins=50)
            if hist[25] <= 0:
                return True
        return False


size = 1
start_x = -7
start_y = -11
amount_x = 15
amount_y = 22

pixels = {}

results = np.zeros((amount_x, amount_y))
index = 0
for y in range(amount_y):
    bottom = start_y + size*y
    for x in range(amount_x):
        left = start_x + size*x
        b = Box(spike_info, left, bottom, size, index)
        pixels[(b.left, b.bottom)] = b
        index += 1
        if b.is_single_unit:
            results[x, y] = 1

r = np.fliplr(results).transpose()


test_x = 0
test_y = 0
test = GroupOfBoxes(pixels[(test_x, test_y)])
test = GroupOfBoxes(Box(spike_info, -6, -8, 4, 0))
for y in range(amount_y):
    bottom = start_y + size*y
    for x in range(amount_x):
        left = start_x + size*x
        if test_x != left and test_y != bottom:
            b = Box(spike_info, left, bottom, size, 0)
            if test.check_single_unit_with_extra_box(b):
                test.add_box(b)

t = np.zeros((amount_x, amount_y))
for y in range(amount_y):
    bottom = start_y + size*y
    for x in range(amount_x):
        left = start_x + size*x
        try:
            p = test.boxes[(left, bottom)]
        except KeyError:
            t[x, y] = 0
        else:
            t[x, y] = 1
t = np.fliplr(t).transpose()





start_x = -7
start_y = -8
step = 0.2
end_x = 7
end_y = 9

def find_single_units(spike_info, start_x, step, end_x, top, bottom):
    left = start_x
    right = start_x + step

    single_units = []
    while right < end_x:
        spikes = find_spikes_in_rect(spike_info, left, right, top, bottom)
        times = np.array(spike_info['times'].iloc[spikes].tolist())
        if is_single_unit(times):
            right += step
        else:
            spikes = find_spikes_in_rect(spike_info, left, right - step, top, bottom)
            times = np.array(spike_info['times'].iloc[spikes].tolist())
            if len(times) > 1000:
                single_units.append(spikes)
            left = right + step
            right = left + step
    single_units = np.array(single_units)

    return single_units


def create_new_spike_info():
    spike_info = np.load(join(tsne_folder, 'template_{}'.format(mua_template), 'spike_info_original.df'))
    spike_info_for_labels = spike_info.copy()
    for index in range(len(single_units)):
        single_unit = single_units[index]
        spike_info_for_labels[ct.TEMPLATE_AFTER_SORTING].iloc[single_unit] = mua_template + 3000 + index
        spike_info_for_labels[ct.TYPE_AFTER_SORTING].iloc[single_unit] = ct.types[1]
    spike_info_for_labels.to_pickle(join(tsne_folder, 'template_{}'.format(mua_template), 'spike_info.df'))

    return spike_info_for_labels


single_units = find_single_units(spike_info, start_x=-7, step=0.2, end_x=8, top=9, bottom=-8)
spike_info_for_labels = create_new_spike_info()
#spike_info.to_pickle(join(tsne_folder, 'template_{}'.format(mua_template), 'spike_info_original.df'))


vis.plot_tsne_of_spikes(spike_info_for_labels)

tsne = np.array([spike_info['tsne_x'].values, spike_info['tsne_y'].values])
''' 
pca = PCA(n_components=1)
pca.fit(tsne)
tsne_rot = pca.transform(tsne)
tsne_rot = [tsne_rot[0] - np.mean(tsne_rot[0]), tsne_rot[1] - np.mean(tsne_rot[1])]
'''


degrees = 20
theta = (degrees/360) * 2*np.pi

# Fit the x y tsne to find a trend
p = np.polyfit(spike_info['tsne_x'].values, spike_info['tsne_y'].values, deg=1)
theta = -np.arcsin(p[0])

rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
tsne_rot = np.dot(rot_matrix, tsne)
plt.scatter(tsne[0], tsne[1])
plt.scatter(tsne_rot[0], tsne_rot[1])

spike_info_rot = spike_info.copy()
spike_info_rot['tsne_x'] = tsne_rot[0]
spike_info_rot['tsne_y'] = tsne_rot[1]

single_units = find_single_units(spike_info_rot, start_x=-12, step=0.2, end_x=12, top=12, bottom=-12)
spike_info_for_labels = create_new_spike_info()
vis.plot_tsne_of_spikes(spike_info_for_labels)


def size_of_single_units():
    i = 0
    for s in single_units:
        i += len(s)
    return i

size_of_single_units()






ss = 0
sc = 0
sp = 0
for t in large_mua_templates[:14]:
    spike_info = preproc_kilo.load_spike_info_of_template(tsne_folder, t)
    ss += len(np.unique(spike_info['template_after_sorting'][spike_info['type_after_sorting']=='SS']))
    sc += len(np.unique(spike_info['template_after_sorting'][spike_info['type_after_sorting'] == 'SS_Contaminated']))
    sp += len(np.unique(spike_info['template_after_sorting'][spike_info['type_after_sorting'] == 'SS_Putative']))


# Stuff for working with the individual mua t-snes that come out of using only the cleaned templates for the
# template_features_matrix (the one that looks like strings)
'''
tsne_folder = join(tsne_folder, 'Single_MUA_templates_Bug')
spike_info = preproc_kilo.load_spike_info_of_template(tsne_folder, mua_template)

this_mua_folder = join(tsne_folder, 'template_{}'.format(large_mua_templates[mua_template_to_look]))
# spike_info.to_pickle(join(this_mua_folder, 'spike_info_original.df'))



vis.make_video_of_tsne_iterations(spike_info, iterations=3000, video_dir=this_mua_folder,
                                  data_file_name='interim_{:0>6}.dat',
                                  video_file_name='tsne_video.mp4', figsize=(15, 15), dpi=200, fps=30,
                                  movie_metadata=None, cm=None, subtitle=None, label_name='Label',
                                  legent_on=False, max_screen=False)


# DBSCAN
data = np.array([spike_info['tsne_x'], spike_info['tsne_y']])
db, n_clusters, labels, core_samples_mask, score = tsne_funcs.fit_dbscan(data, eps=0.029, min_samples=43, normalize=True,
                                                                         show=True)
spike_info_for_labels = spike_info.copy()
spike_info_for_labels[ct.TEMPLATE_AFTER_CLEANING] = labels
spike_info_for_labels[ct.TEMPLATE_AFTER_SORTING] = labels
spike_info_for_labels.to_pickle(join(tsne_folder, 'template_{}'.format(large_mua_templates[mua_template_to_look]),
                                     'spike_info_for_labels.df'))
spike_info_for_labels.to_pickle(join(tsne_folder, 'template_{}'.format(large_mua_templates[mua_template_to_look]),
                                     'spike_info.df'))



num_of_spikes_per_cluster = []
histograms = []
good_histograms = {}

clusters_in_size_order = []
for cluster in range(n_clusters):
    indices_of_spikes = np.squeeze(np.argwhere(labels == cluster))
    clusters_in_size_order.append(cluster)
    num_of_spikes_per_cluster.append(len(indices_of_spikes))
    if len(indices_of_spikes) > 500:
        spike_times = np.array(spike_info['times'].iloc[indices_of_spikes].tolist())
        diffs, norm = hf.crosscorrelate_spike_trains(spike_times, spike_times, lag=1000)
        hist, edges = np.histogram(diffs, bins=50)
        histograms.append(hist)
        if hist[25] == 0:
            good_histograms[cluster] = hist

histograms = np.array(histograms)

clusters_in_size_order = [x for _, x in sorted(zip(num_of_spikes_per_cluster, clusters_in_size_order))]
clusters_in_size_order.reverse()

merged_clusters = {}
merged_clusters_size = {}
clusters_to_ignore = []
for cluster in clusters_in_size_order:
    merged_clusters[cluster] = [cluster]
    clusters_to_ignore.append(cluster)
    indices_of_spikes = np.squeeze(np.argwhere(labels == cluster))
    spike_times = np.array(spike_info['times'].iloc[indices_of_spikes].tolist())
    clusters_to_try_to_merge = clusters_in_size_order.copy()
    for ignore in clusters_to_ignore:
        clusters_to_try_to_merge.remove(ignore)
    for cluster_to_check in clusters_to_try_to_merge:
        indices_of_spikes_to_check = np.squeeze(np.argwhere(labels == cluster_to_check))
        spike_times_to_check = np.array(spike_info['times'].iloc[indices_of_spikes_to_check].tolist())
        test_spike_times = np.concatenate((spike_times, spike_times_to_check))
        diffs, norm = hf.crosscorrelate_spike_trains(test_spike_times, test_spike_times, lag=1000)
        hist, edges = np.histogram(diffs, bins=50)
        if hist[25] <= 0:
            merged_clusters[cluster].append(cluster_to_check)
            spike_times = test_spike_times
    merged_clusters_size[cluster] = len(spike_times)
    print('Finished cluster {}'.format(cluster))


import sequence_viewer as seq_view
import one_shot_viewer as osv

h = 0

osv.graph(globals(), 'hist', 'edges')
seq_view.graph_pane(globals(), 'h', 'histograms', 'edges')
'''