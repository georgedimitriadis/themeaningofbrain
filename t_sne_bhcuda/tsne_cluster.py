
from bokeh.client import push_session
from bokeh.layouts import column, row, layout
from bokeh.models import BoxSelectTool, LassoSelectTool, ColumnDataSource, TextInput, Paragraph, VBox, HBox, Circle, \
CustomJS
from bokeh.plotting import curdoc, figure, reset_output
from bokeh.models.widgets import DataTable, TableColumn, Button, Select, Toggle, CheckboxGroup, Slider
import numpy as np
import pandas as pd
from t_sne_bhcuda import spike_heatmap
from itertools import chain
import copy

# globals
previous_tsne_source_selected = None
previously_selected_spike_indices = []
currently_selected_spike_indices = []
update_spike_corr_heatmap_figs = True
tsne_clusters_scatter_plot = None
checkbox_show_clusters_with_colors = None
non_selected_points_alpha = 0.2
selected_points_size = 2
non_selected_points_size = 2
selected_cluster_names = None
num_of_spikes_used = 0
clusters_of_all_spikes = []
update_old_selected_switch = True

def gui_manual_cluster_tsne_spikes(tsne_array_or_filename, spike_times_list_or_filename, raw_extracellular_data,
                                   num_of_points_for_baseline, cut_extracellular_data_or_filename,
                                   shape_of_cut_extracellular_data, cube_type,
                                   sampling_freq, autocor_bin_number,
                                     cluster_info_file, use_existing_cluster=False, time_samples_h5_dir=None,
                                   spike_indices_to_use=None, prb_file=None, k4=False,
                                   verbose=False):
    """
    Creates a GUI that shows the t-sne data and allows selection of them, showing the average spikes forms, the
    autocorrelogram and the heatmap of the selected spikes. It also allows putting the selected spikes in clusters.
    Either the raw extracellular data or the cut extracellular data cube must have the same number of spikes as the
    provided t-sne data set. A sub-selection of spikes to be shown on the t-sne gui can be done by providing the
    spike_indices_to_use array. But the provided data (voltage and t-sne) must have the same number of spikes
    to begin with and their indices must correspond since this is not checked in this method.
    If the number of spikes in the t-sne data and the length of the spike_indices_to_use array do not match then the
    cut extracellular data cube will be recreated even if the cut_extracellular_data_or_filename parameters points to
    an existing file.

    ATTENTION:
    FOR THE GUI TO WORK A BOKEH SERVER MUST BE RUNNING!!

    To start a bokeh server (assuming the bokeh.exe is in your path) do bokeh serve in a command promt (or check the
    bokeh manual).

    Parameters
    ----------
    tsne_array_or_filename: filename of t-sne data or t-sne data array (2 x spike number)
    spike_times_list_or_filename: the .kwik filename where the spike times are saved or an array with the spike times
    raw_extracellular_data: the raw array (electrodes x time points) of the recording
    num_of_points_for_baseline: number of time points to use to baseline the spike traces (no filtering is used so some
    baselining helps).
    cut_extracellular_data_or_filename: instead of the raw_extracellular_data one can pass either the .npy filename
    holding an electrodes x time points x spikes already cut data cube or the data cube itself. The function will use
    (or create) the .npy file to memmap (thus not loading the full cut data on RAM).
    shape_of_cut_extracellular_data: the number of electrodes x number of time points in a spike x number of spikes
    cube_type: the data type of the cut cube data
    sampling_freq: the sampling frequency of the recording
    autocor_bin_number: the amount of bins that the autocorellogram will be split into
    cluster_info_file: the file name to hold the cluster info (a .pkl file since this is a pickle of a pandas dataframe)
    use_existing_cluster: if False then any .pkl file with the same name will be overwritten and there will be no
    cluster info at the beginning of the GUI. If True then the existing .pkl file will be expanded (and its clusters
    will appear at the beginning of the GUI).
    time_samples_h5_dir: the directory structure within the .kwik file where the spike times are saved
    spike_indices_to_use: A sub-sample of the spikes passed by the t-sne data and the raw or cut extracellular data
    prb_file: the probe geometry file defining the probe as used in the phy module
    k4: if True the screen is 4k (defines the size of the gui). Otherwise it is assumed to be HD
    verbose: if True then the GUI will print info on the interpreter

    Returns
    -------

    """
    global num_of_spikes_used
    global clusters_of_all_spikes

    if type(tsne_array_or_filename) is str:
        tsne_array_or_filename = get_tsne_from_filename(tsne_array_or_filename)

    # Select the tsne samples to use
    num_of_initial_spikes = np.shape(tsne_array_or_filename)[1]
    if spike_indices_to_use is None:
        spike_indices_to_use = np.arange(num_of_initial_spikes)
    tsne_array_or_filename = tsne_array_or_filename[:, spike_indices_to_use]

    if type(spike_times_list_or_filename) is str and time_samples_h5_dir is not None:
        used_extra_spike_times = get_spike_times_from_kwik_file(spike_times_list_or_filename, time_samples_h5_dir,
                                                                verbose=verbose)
    elif type(spike_times_list_or_filename) is str and time_samples_h5_dir is None:
        print('If the .kwik filename is given, the hdf5 file internal directory where the spike times are, ' +
              'must also be given')
        return
    else:
        used_extra_spike_times = spike_times_list_or_filename

    used_extra_spike_times = used_extra_spike_times[spike_indices_to_use]
    num_of_spikes_used = len(used_extra_spike_times)

    if num_of_spikes_used != shape_of_cut_extracellular_data[2]:
        print('The number of chosen spikes in spike_indices_to_use and the number of spikes in ' +
              'shape_of_cut_extracellular_data[2] must be equal')
        return

    num_of_points_in_spike_trig = shape_of_cut_extracellular_data[1]

    # Generate a new data cube file if raw data are provided and the user passes a filename to save the memmap
    # of the cut cube in
    # If a new data cube is not generated then load the existing one (with the given filename). Warn the user that the
    # loaded data cube needs to have a number of spikes equal to the selected ones (if there is subselection of spikes)
    if raw_extracellular_data is not None and type(cut_extracellular_data_or_filename) is str:
        print('Recreating the channels x time x spikes data cube')
        num_ivm_channels = raw_extracellular_data.shape[0]
        if num_ivm_channels != shape_of_cut_extracellular_data[0]:
            print('The number of channels in the raw_extracellular_data and the number of channels in ' +
                  'shape_of_cut_extracellular_data[0] must be equal')
            return
        cut_extracellular_data = create_data_cube_from_raw_extra_data(raw_extracellular_data,
                                                                      cut_extracellular_data_or_filename,
                                                                      num_ivm_channels, num_of_points_in_spike_trig,
                                                                      cube_type, used_extra_spike_times,
                                                                      num_of_points_for_baseline=
                                                                      num_of_points_for_baseline)
    elif cut_extracellular_data_or_filename is not None and type(cut_extracellular_data_or_filename) is not str:
        cut_extracellular_data = cut_extracellular_data_or_filename
    else:
        if num_of_spikes_used != num_of_initial_spikes:
            print('Warning! If the cut_extracellular_data_or_filename does not point to a cut data cube with a number' +
                  'of spikes equal to the selected ones the gui will fail!')
        import os.path as path
        if type(cut_extracellular_data_or_filename) is str and path.isfile(cut_extracellular_data_or_filename):
            print('Loading an existing channels x time x spikes data cube')
            cut_extracellular_data = load_extracellular_data_cube(cut_extracellular_data_or_filename, cube_type,
                                                                  shape_of_cut_extracellular_data)
        else:
            print("If no extracellular raw or cut data are provided then a filename pointing to the " +
                  "cut extra data cube should be given")
            return

    if type(cluster_info_file) is str:
        import os.path as path
        if not path.isfile(cluster_info_file):
            create_new_cluster_info_file(cluster_info_file, num_of_spikes_used)


    time_axis = generate_time_axis(num_of_points_in_spike_trig, sampling_freq)
    clusters_of_all_spikes = np.empty((num_of_spikes_used), dtype=np.int32)

    generate_gui(tsne_array_or_filename, cut_extracellular_data, used_extra_spike_times, time_axis,
                 cluster_info_file, use_existing_cluster, autocor_bin_number, sampling_freq, prb_file, k4, verbose)


# Data generation and manipulation functions
def get_tsne_from_filename(tsne_filename):
    return np.load(tsne_filename)


def get_spike_times_from_kwik_file(kwik_filename, time_samples_h5_dir, verbose=False):
    import h5py as h5
    h5file = h5.File(kwik_filename, mode='r')
    all_extra_spike_times = np.array(list(h5file[time_samples_h5_dir]))
    h5file.close()
    if verbose:
        print('All extra spikes = {}'.format(len(all_extra_spike_times)))
    return all_extra_spike_times


def create_data_cube_from_raw_extra_data(raw_extracellular_data, data_cube_filename,
                                         num_of_points_in_spike_trig, cube_type, extra_spike_times,
                                         num_of_electrodes=None, used_electrodes=None,
                                         num_of_points_for_baseline=None):
    import os.path as path
    if path.isfile(data_cube_filename):
        import os
        os.remove(data_cube_filename)

    num_of_spikes = len(extra_spike_times)
    if used_electrodes is None and num_of_electrodes is not None:
        used_electrodes = np.arange(num_of_electrodes)
    elif used_electrodes is not None:
        num_of_electrodes = used_electrodes.shape[0]
    else:
        print('Please provide either a number of electrodes or a list of electrodes')
        return
    shape_of_spike_trig_avg = ((num_of_electrodes,
                                num_of_points_in_spike_trig,
                                num_of_spikes))

    data_cube = np.memmap(data_cube_filename,
                          dtype=cube_type,
                          mode='w+',
                          shape=shape_of_spike_trig_avg)
    for spike in np.arange(0, num_of_spikes):
        trigger_point = extra_spike_times[spike]
        start_point = int(trigger_point - num_of_points_in_spike_trig / 2)
        if start_point < 0:
            break
        end_point = int(trigger_point + num_of_points_in_spike_trig / 2)
        if end_point > raw_extracellular_data.shape[1]:
            break
        temp = raw_extracellular_data[used_electrodes, start_point:end_point]
        if num_of_points_for_baseline is not None:
            baseline = np.mean(temp[:, [0, num_of_points_for_baseline]], 1)
            temp = (temp.T - baseline.T).T
        data_cube[:, :, spike] = temp.astype(cube_type)
        if spike % 1000 == 0:
            print('Done ' + str(spike) + ' spikes')
        del temp
    del raw_extracellular_data
    del baseline
    del data_cube

    cut_extracellular_data = load_extracellular_data_cube(data_cube_filename, cube_type, shape_of_spike_trig_avg)

    return cut_extracellular_data


def load_extracellular_data_cube(data_cube_filename, cube_type,
                                 shape_of_spike_trig_avg):
    cut_extracellular_data = np.memmap(data_cube_filename,
                                       dtype=cube_type,
                                       mode='r',
                                       shape=shape_of_spike_trig_avg)
    return cut_extracellular_data


def generate_time_axis(num_of_points_in_spike_trig, sampling_freq):
    time_axis = np.arange(-num_of_points_in_spike_trig/(2*sampling_freq),
                          num_of_points_in_spike_trig/(2*sampling_freq),
                          1/sampling_freq)
    return time_axis


# Spike train autocorelogram
def crosscorrelate_spike_trains(spike_times_train_1, spike_times_train_2, lag=None):
    if spike_times_train_1.size < spike_times_train_2.size:
        if lag is None:
            lag = np.ceil(10 * np.mean(np.diff(spike_times_train_1)))
        reverse = False
    else:
        if lag is None:
            lag = np.ceil(20 * np.mean(np.diff(spike_times_train_2)))
        spike_times_train_1, spike_times_train_2 = spike_times_train_2, spike_times_train_1
        reverse = True

    # calculate cross differences in spike times
    differences = np.array([])
    for k in np.arange(0, spike_times_train_1.size):
        differences = np.append(differences, spike_times_train_1[k] - spike_times_train_2[np.nonzero(
            (spike_times_train_2 > spike_times_train_1[k] - lag)
             & (spike_times_train_2 < spike_times_train_1[k] + lag)
             & (spike_times_train_2 != spike_times_train_1[k]))])
    if reverse is True:
        differences = -differences
    norm = np.sqrt(spike_times_train_1.size * spike_times_train_2.size)
    return differences, norm


# Cluster info file and pandas series functions
def create_new_cluster_info_file(filename, tsne_length):
    cluster_info = pd.DataFrame(
        {'Cluster': 'UNLABELD', 'Num_of_Spikes': tsne_length, 'Spike_Indices': [np.arange(tsne_length)]})
    cluster_info = cluster_info.append(pd.Series({'Cluster': 'NOISE', 'Num_of_Spikes': 0, 'Spike_Indices': []}),
                                       ignore_index=True)
    cluster_info = cluster_info.append(pd.Series({'Cluster': 'MUA', 'Num_of_Spikes': 0, 'Spike_Indices': []}),
                                       ignore_index=True)
    cluster_info = cluster_info.set_index('Cluster')
    cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)
    cluster_info.to_pickle(filename)
    return cluster_info


# This is not used in the gui but it is useful to create the cluster_info DataFrame that the gui needs if one has
# the kilosort spike_templates.npy file
def create_cluster_info_from_kilosort_spike_templates(cluster_info_filename, spike_templates):
    kilosort_units = {}
    for i in np.arange(len(spike_templates)):
        cluster = spike_templates[i][0]
        if cluster in kilosort_units:
            kilosort_units[cluster] = np.append(kilosort_units[cluster], i)
        else:
            kilosort_units[cluster] = i

    cluster_info = pd.DataFrame(columns=['Cluster', 'Num_of_Spikes', 'Spike_Indices'])
    cluster_info = cluster_info.set_index('Cluster')
    cluster_info['Spike_Indices'] = cluster_info['Spike_Indices'].astype(list)

    cluster_info.set_value('UNLABELED', 'Num_of_Spikes', 0)
    cluster_info.set_value('UNLABELED', 'Spike_Indices', [])
    cluster_info.set_value('NOISE', 'Num_of_Spikes', 0)
    cluster_info.set_value('NOISE', 'Spike_Indices', [])
    cluster_info.set_value('MUA', 'Num_of_Spikes', 0)
    cluster_info.set_value('MUA', 'Spike_Indices', [])
    for g in kilosort_units.keys():
        if np.size(kilosort_units[g]) == 1:
            kilosort_units[g] = [kilosort_units[g]]
        cluster_name = str(g)
        cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(kilosort_units[g]))
        cluster_info.set_value(cluster_name, 'Spike_Indices', kilosort_units[g])

    cluster_info.to_pickle(cluster_info_filename)
    return cluster_info

def load_cluster_info(filename):
    global num_of_spikes_used
    global clusters_of_all_spikes
    cluster_info = pd.read_pickle(filename)

    for cluster_str in cluster_info.index:
        spike_indices_in_cluster = set(cluster_info.loc[cluster_str].Spike_Indices)
        cluster_index = cluster_info.index.get_loc(cluster_str)
        for spike_index in spike_indices_in_cluster:
            clusters_of_all_spikes[spike_index] = cluster_index
    return cluster_info


def add_cluster_to_cluster_info(filename, cluster_name, spike_indices):
    cluster_info = load_cluster_info(filename)
    cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(spike_indices))
    cluster_info.set_value(cluster_name, 'Spike_Indices', spike_indices)
    cluster_info.to_pickle(filename)
    cluster_index = cluster_info.index.get_loc(cluster_name)

    return cluster_info


def remove_cluster_from_cluster_info(filename, cluster_name, unassign=True):
    global clusters_of_all_spikes
    cluster_info = load_cluster_info(filename)

    spike_indices = cluster_info.loc[cluster_name].Spike_Indices
    cluster_index = cluster_info.index.get_loc(cluster_name)

    if unassign:
        cluster_info = add_spikes_to_cluster(filename, 'UNLABELED', spike_indices)

    cluster_info = cluster_info.drop(cluster_name)
    cluster_info.to_pickle(filename)
    return cluster_info


def add_spikes_to_cluster(filename, cluster_name, spike_indices):
    global clusters_of_all_spikes
    cluster_info = load_cluster_info(filename)
    new_indices = np.append(cluster_info['Spike_Indices'][cluster_name], spike_indices).astype(np.int32)
    cluster_info.set_value(cluster_name, 'Spike_Indices', new_indices)
    cluster_info.set_value(cluster_name, 'Num_of_Spikes', len(new_indices))
    cluster_info.to_pickle(filename)
    cluster_index = cluster_info.index.get_loc(cluster_name)
    for spike_index in spike_indices:
        clusters_of_all_spikes[spike_index] = cluster_index
    return cluster_info

def remove_spikes_from_all_clusters(filename, spike_indices):
    global clusters_of_all_spikes
    spike_indices_in_used_clusters = {}
    cluster_info = load_cluster_info(filename)

    for spike_index in spike_indices:
        cluster_index = clusters_of_all_spikes[spike_index]
        if cluster_index not in spike_indices_in_used_clusters.keys():
            spike_indices_in_used_clusters[cluster_index] = [spike_index]
        else:
            spike_indices_in_used_clusters[cluster_index].append(spike_index)

    if len(spike_indices_in_used_clusters) > 0:
        for cluster_index in spike_indices_in_used_clusters.keys():
            cluster_name = cluster_info.index[cluster_index]
            cluster_info = remove_spikes_from_cluster(filename, cluster_name,
                                                      spike_indices_in_used_clusters[cluster_index], unassign=False)
    return cluster_info


def remove_spikes_from_cluster(filename, cluster_name, spike_indices, unassign=True):
    global clusters_of_all_spikes
    cluster_info = load_cluster_info(filename)

    if len(cluster_info['Spike_Indices'][cluster_name]) == len(spike_indices):
        remove_cluster_from_cluster_info(filename, cluster_name, unassign=False)
    else:
        mask = np.in1d(cluster_info['Spike_Indices'][cluster_name], spike_indices)
        new_indices = cluster_info['Spike_Indices'][cluster_name][~mask]
        cluster_info = add_cluster_to_cluster_info(filename, cluster_name, new_indices)
        cluster_index = cluster_info.index.get_loc(cluster_name)

        if unassign:
            add_spikes_to_cluster(filename, 'UNLABELED', spike_indices)
            for spike_index in spike_indices:
                if clusters_of_all_spikes[spike_index] == cluster_index:
                    clusters_of_all_spikes[spike_index] = 0

    return cluster_info

def remove_spikes_from_u_m_n(filename, spike_indices):
    global clusters_of_all_spikes
    cluster_info = load_cluster_info(filename)
    for spike_index in spike_indices:
        cluster_index = clusters_of_all_spikes[spike_index]
        if cluster_index == 0:
            cluster_info.loc['UNLABELED'].Spike_Indices = \
                cluster_info.loc['UNLABELED'].Spike_Indices[(cluster_info.loc['UNLABELED'].Spike_Indices != spike_index)]
        if cluster_index == 1:
            cluster_info.loc['MUA'].Spike_Indices = \
                cluster_info.loc['MUA'].Spike_Indices[(cluster_info.loc['UNLABELED'].Spike_Indices != spike_index)]
        if cluster_index == 2:
            cluster_info.loc['NOISE'].Spike_Indices = \
                cluster_info.loc['NOISE'].Spike_Indices[(cluster_info.loc['UNLABELED'].Spike_Indices != spike_index)]
    cluster_info.to_pickle(filename)

    return cluster_info

# Gui generator
def generate_gui(tsne, cut_extracellular_data, all_extra_spike_times, time_axis, cluster_info_file,
                 use_existing_cluster, autocor_bin_number, sampling_freq, prb_file=None, k4=False, verbose=False):

    if k4:
        tsne_figure_size = [1000, 800]
        tsne_min_border_left = 50
        spike_figure_size = [500, 500]
        hist_figure_size = [500, 500]
        heatmap_plot_size = [200, 800]
        clusters_table_size = [400, 300]
        layout_size = [1500, 1400]
        slider_size = [300, 100]
        user_info_size = [700, 80]
    else:
        tsne_figure_size = [850, 600]
        tsne_min_border_left = 10
        spike_figure_size = [450, 300]
        hist_figure_size = [450, 300]
        heatmap_plot_size = [200, 800]
        clusters_table_size = [400, 400]
        layout_size = [1200, 800]
        slider_size = [270, 80]
        user_info_size = [450, 80]
    # Plots ------------------------------
    # scatter plot
    global non_selected_points_alpha
    global selected_points_size
    global non_selected_points_size
    global update_old_selected_switch
    global previously_selected_spike_indices

    tsne_fig_tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,tap,resize,reset,save"
    tsne_figure = figure(tools=tsne_fig_tools, plot_width=tsne_figure_size[0], plot_height=tsne_figure_size[1],
                         title='T-sne', min_border=10, min_border_left=tsne_min_border_left, webgl=True)

    tsne_source = ColumnDataSource({'tsne-x': tsne[0], 'tsne-y': tsne[1]})

    tsne_selected_points_glyph = Circle(x='tsne-x', y='tsne-y', size=selected_points_size,
                                        line_alpha=0, fill_alpha=1, fill_color='red')
    tsne_nonselected_points_glyph = Circle(x='tsne-x', y='tsne-y', size=non_selected_points_size,
                                           line_alpha=0, fill_alpha=non_selected_points_alpha, fill_color='blue')
    tsne_invisible_points_glyph = Circle(x='tsne-x', y='tsne-y', size=selected_points_size, line_alpha=0, fill_alpha=0)

    tsne_nonselected_glyph_renderer = tsne_figure.add_glyph(tsne_source, tsne_nonselected_points_glyph,
                                                            selection_glyph=tsne_invisible_points_glyph,
                                                            nonselection_glyph=tsne_nonselected_points_glyph,
                                                            name='tsne_nonselected_glyph_renderer')
        # note: the invisible glyph is required to be able to change the size of the selected points, since the
        # use of selection_glyph is usefull only for colors and alphas
    tsne_invinsible_glyph_renderer = tsne_figure.add_glyph(tsne_source, tsne_invisible_points_glyph,
                                                           selection_glyph=tsne_selected_points_glyph,
                                                           nonselection_glyph=tsne_invisible_points_glyph,
                                                           name='tsne_invinsible_glyph_renderer')


    tsne_figure.select(BoxSelectTool).select_every_mousemove = False
    tsne_figure.select(LassoSelectTool).select_every_mousemove = False


    def on_tsne_data_update(attr, old, new):
        global previously_selected_spike_indices
        global currently_selected_spike_indices
        global non_selected_points_alpha
        global non_selected_points_size
        global selected_points_size
        global checkbox_find_clusters_of_selected_points

        previously_selected_spike_indices = np.array(old['1d']['indices'])
        currently_selected_spike_indices = np.array(new['1d']['indices'])
        num_of_selected_spikes = len(currently_selected_spike_indices)

        if num_of_selected_spikes > 0:
            if verbose:
                print('Num of selected spikes = ' + str(num_of_selected_spikes))

            # update t-sne plot
            tsne_invisible_points_glyph.size = selected_points_size
            tsne_nonselected_points_glyph.size = non_selected_points_size
            tsne_nonselected_points_glyph.fill_alpha = non_selected_points_alpha

            # update spike plot
            avg_x = np.mean(cut_extracellular_data[:, :, currently_selected_spike_indices], axis=2)
            spike_mline_plot.data_source.data['ys'] = avg_x.tolist()
            print('Finished avg spike plot')

            # update autocorelogram
            diffs, norm = crosscorrelate_spike_trains(all_extra_spike_times[currently_selected_spike_indices].astype(np.int64),
                                                      all_extra_spike_times[currently_selected_spike_indices].astype(np.int64), lag=1500)
            hist, edges = np.histogram(diffs, bins=autocor_bin_number)
            hist_plot.data_source.data["top"] = hist
            hist_plot.data_source.data["left"] = edges[:-1] / sampling_freq
            hist_plot.data_source.data["right"] = edges[1:] / sampling_freq
            print('finished autocorelogram')

            # update heatmap
            if prb_file is not None:
                print('Doing heatmap')
                data = cut_extracellular_data[:, :, currently_selected_spike_indices]
                final_image, (x_size, y_size) = spike_heatmap.create_heatmap(data, prb_file, rotate_90=True,
                                                                             flip_ud=True, flip_lr=False)
                new_image_data = dict(image=[final_image], x=[0], y=[0], dw=[x_size], dh=[y_size])
                heatmap_data_source.data.update(new_image_data)
                print('Finished heatmap')

    tsne_source.on_change('selected', on_tsne_data_update)

    # spike plot
    spike_fig_tools = 'pan,wheel_zoom,box_zoom,reset,save'
    spike_figure = figure(toolbar_location='below', plot_width=spike_figure_size[0], plot_height=spike_figure_size[1],
                          tools=spike_fig_tools, title='Spike average', min_border=10, webgl=True, toolbar_sticky=False)

    num_of_channels = cut_extracellular_data.shape[0]
    num_of_time_points = cut_extracellular_data.shape[1]
    xs = np.repeat(np.expand_dims(time_axis, axis=0), repeats=num_of_channels, axis=0).tolist()
    ys = np.ones((num_of_channels, num_of_time_points)).tolist()
    spike_mline_plot = spike_figure.multi_line(xs=xs, ys=ys)

    # autocorelogram plot
    hist, edges = np.histogram([], bins=autocor_bin_number)
    hist_fig_tools = 'pan,wheel_zoom,box_zoom,save,reset'

    hist_figure = figure(toolbar_location='below', plot_width=hist_figure_size[0], plot_height=hist_figure_size[1],
                         tools=hist_fig_tools, title='Autocorrelogram', min_border=10, webgl=True, toolbar_sticky=False)
    hist_plot = hist_figure.quad(bottom=0, left=edges[:-1], right=edges[1:], top=hist, color="#3A5785", alpha=0.5,
                                 line_color="#3A5785")
    # heatmap plot
    heatmap_plot = figure(toolbar_location='right', plot_width=1, plot_height=heatmap_plot_size[1],
                          x_range=(0, 1), y_range=(0, 1), title='Probe heatmap',
                          toolbar_sticky=False)
    if prb_file is not None:
        data = np.zeros(cut_extracellular_data.shape)
        final_image, (x_size, y_size) = spike_heatmap.create_heatmap(data, prb_file, rotate_90=True,
                                                                     flip_ud=True, flip_lr=False)
        final_image[:, :, ] = 4294967295  # The int32 for the int8 255 (white)
        plot_width = max(heatmap_plot_size[0], int(heatmap_plot_size[1] * y_size / x_size))
        heatmap_plot = figure(toolbar_location='right', plot_width=plot_width, plot_height=heatmap_plot_size[1],
                              x_range=(0, x_size), y_range=(0, y_size), title='Probe heatmap',
                              toolbar_sticky=False)

        heatmap_data_source = ColumnDataSource(data=dict(image=[final_image], x=[0], y=[0], dw=[x_size], dh=[y_size]))
        heatmap_renderer = heatmap_plot.image_rgba(source=heatmap_data_source, image='image', x='x', y='y',
                                                   dw='dw', dh='dh', dilate=False)
        heatmap_plot.axis.visible = None
        heatmap_plot.xgrid.grid_line_color = None
        heatmap_plot.ygrid.grid_line_color = None
    # ---------------------------------------
    # --------------- CONTROLS --------------
    # Texts and Tables
    # the clusters DataTable
    if use_existing_cluster:
        cluster_info = load_cluster_info(cluster_info_file)
    else:
        cluster_info = create_new_cluster_info_file(cluster_info_file, len(tsne))
    cluster_info_data_source = ColumnDataSource(cluster_info)
    clusters_columns = [TableColumn(field='Cluster', title='Clusters'),
                        TableColumn(field='Num_of_Spikes', title='Number of Spikes')]
    clusters_table = DataTable(source=cluster_info_data_source, columns=clusters_columns, selectable=True,
                               editable=False, width=clusters_table_size[0], height=clusters_table_size[1],
                               scroll_to_selection=True)

    def on_select_cluster_info_table(attr, old, new):
        global selected_cluster_names
        cluster_info = load_cluster_info(cluster_info_file)
        indices = list(chain.from_iterable(cluster_info.iloc[new['1d']['indices']].Spike_Indices.tolist()))
        selected_cluster_names = list(cluster_info.index[new['1d']['indices']])
        old = new = tsne_source.selected
        tsne_source.selected['1d']['indices'] = indices
        tsne_source.trigger('selected', old, new)
        user_info_edit.value = 'Selected clusters = ' + ', '.join(selected_cluster_names)

    cluster_info_data_source.on_change('selected', on_select_cluster_info_table)

    def update_data_table():
        cluster_info_data_source = ColumnDataSource(load_cluster_info(cluster_info_file))
        cluster_info_data_source.on_change('selected', on_select_cluster_info_table)
        clusters_table.source = cluster_info_data_source
        options = list(cluster_info_data_source.data['Cluster'])
        options.insert(0, 'No cluster selected')
        select_cluster_to_move_points_to.options = options

    # cluster TextBox that adds cluster to the DataTable
    new_cluster_name_edit = TextInput(value='give the new cluster a name',
                                      title='Put selected points into a new cluster')

    def on_text_edit_new_cluster_name(attr, old, new):
        global currently_selected_spike_indices
        global clusters_of_all_spikes

        new_cluster_name = new_cluster_name_edit.value

        spike_indices_to_delete_from_existing_clusters = {}
        for spike_index in currently_selected_spike_indices:
            if clusters_of_all_spikes[spike_index] != -1:
                cluster_index = clusters_of_all_spikes[spike_index]
                if cluster_index not in spike_indices_to_delete_from_existing_clusters:
                    spike_indices_to_delete_from_existing_clusters[cluster_index] = [spike_index]
                else:
                    spike_indices_to_delete_from_existing_clusters[cluster_index].append(spike_index)
        cluster_info = load_cluster_info(cluster_info_file)
        for cluster_index in spike_indices_to_delete_from_existing_clusters.keys():
            cluster_name = cluster_info.iloc[cluster_index].name
            remove_spikes_from_cluster(cluster_info_file, cluster_name,
                                       spike_indices_to_delete_from_existing_clusters[cluster_index], unassign=False)

        add_cluster_to_cluster_info(cluster_info_file, new_cluster_name, currently_selected_spike_indices)

        update_data_table()

    new_cluster_name_edit.on_change('value', on_text_edit_new_cluster_name)

    # user information Text
    user_info_edit = TextInput(value='', title='User information',
                               width=user_info_size[0], height=user_info_size[1])

    # Buttons ------------------------
    # show all clusters Button
    button_show_all_clusters = Toggle(label='Show all clusters', button_type='primary')

    def on_button_show_all_clusters(state, *args):
        global tsne_clusters_scatter_plot

        if state:
            cluster_info = load_cluster_info(cluster_info_file)
            num_of_clusters = cluster_info.shape[0]
            indices_list_of_lists = cluster_info['Spike_Indices'].tolist()
            indices = [item for sublist in indices_list_of_lists for item in sublist]
            cluster_indices = np.arange(num_of_clusters)

            if verbose:
                print('Showing all clusters in colors... wait for it...')

            colors = []
            for c in cluster_indices:
                r = np.random.random(size=1) * 255
                g = np.random.random(size=1) * 255
                for i in np.arange(len(indices_list_of_lists[c])):
                    colors.append("#%02x%02x%02x" % (int(r), int(g), 50))

            first_time = True
            for renderer in tsne_figure.renderers:
                if renderer.name == 'tsne_all_clusters_glyph_renderer':
                    renderer.data_source.data['fill_color'] = renderer.data_source.data['line_color'] = colors
                    renderer.glyph.fill_color = 'fill_color'
                    renderer.glyph.line_color = 'line_color'
                    first_time = False
                    break
            if first_time:
                tsne_clusters_scatter_plot = tsne_figure.scatter(tsne[0][indices], tsne[1][indices], size=1,
                                                                 color=colors, alpha=1,
                                                                 name='tsne_all_clusters_glyph_renderer')
            tsne_clusters_scatter_plot.visible = True
            button_show_all_clusters.label = 'Hide all clusters'
        else:
            if verbose:
                print('Hiding clusters')
            button_show_all_clusters.update()
            tsne_clusters_scatter_plot.visible = False
            button_show_all_clusters.label = 'Show all clusters'

    button_show_all_clusters.on_click(on_button_show_all_clusters)


    # select the clusters that the selected points belong to Button
    # (that will then drive the selection of these spikes on t-sne through the update of the clusters_table source)
    button_show_clusters_of_selected_points = Button(label='Show clusters of selected points')

    def on_button_show_clusters_change():
        print('Hello')
        global clusters_of_all_spikes
        currently_selected_spike_indices = tsne_source.selected['1d']['indices']
        cluster_info = load_cluster_info(cluster_info_file)
        clusters_selected = []
        new_indices_to_select = []
        update_data_table()
        for spike_index in currently_selected_spike_indices:
            if clusters_of_all_spikes[spike_index] not in clusters_selected:
                clusters_selected.append(clusters_of_all_spikes[spike_index])
                indices_in_cluster = cluster_info.iloc[clusters_of_all_spikes[spike_index]].Spike_Indices
                new_indices_to_select.append(indices_in_cluster)
        if len(new_indices_to_select) > 0:
            old = clusters_table.source.selected
            clusters_table.source.selected['1d']['indices'] = clusters_selected
            new = clusters_table.source.selected
            clusters_table.source.trigger('selected', old, new)
            for c in np.arange(len(clusters_selected)):
                clusters_selected[c] = cluster_info.index[clusters_selected[c]]


    button_show_clusters_of_selected_points.on_click(on_button_show_clusters_change)

    # merge clusters Button
    button_merge_clusters_of_selected_points = Button(label='Merge clusters of selected points')

    def on_button_merge_clusters_change():
        global clusters_of_all_spikes
        currently_selected_spike_indices = tsne_source.selected['1d']['indices']
        cluster_info = load_cluster_info(cluster_info_file)
        clusters_selected = []
        for spike_index in currently_selected_spike_indices:
            if clusters_of_all_spikes[spike_index] not in clusters_selected:
                clusters_selected.append(clusters_of_all_spikes[spike_index])
        if len(clusters_selected) > 0:
            clusters_selected = np.sort(clusters_selected)
            clusters_selected_names = []
            for cluster_index in clusters_selected:
                clusters_selected_names.append(cluster_info.iloc[cluster_index].name)
            cluster_name = clusters_selected_names[0]
            add_cluster_to_cluster_info(cluster_info_file, cluster_name, currently_selected_spike_indices)
            i = 0
            for c in np.arange(1, len(clusters_selected)):
                cluster_info = remove_cluster_from_cluster_info(cluster_info_file,
                                                                cluster_info.iloc[clusters_selected[c] - i].name,
                                                                unassign=False)
                i = i + 1 # Every time you remove a cluster the original index of the remaining clusters drops by one

            update_data_table()
            user_info_edit.value = 'Clusters '+ ', '.join(clusters_selected_names) + ' merged to cluster ' + cluster_name

    button_merge_clusters_of_selected_points.on_click(on_button_merge_clusters_change)

    # delete cluster Button
    button_delete_cluster = Button(label='Delete selected cluster(s)')

    def on_button_delete_cluster():
        global selected_cluster_names
        for cluster_name in selected_cluster_names:
            remove_cluster_from_cluster_info(cluster_info_file, cluster_name)
        user_info_edit.value = 'Deleted clusters: ' + ', '.join(selected_cluster_names)
        update_data_table()

    button_delete_cluster.on_click(on_button_delete_cluster)

    # select cluster to move selected points to Select
    select_cluster_to_move_points_to = Select(title="Assign selected points to cluster:", value="No cluster selected")

    options = list(cluster_info_data_source.data['Cluster'])
    options.insert(0, 'No cluster selected')
    select_cluster_to_move_points_to.options = options


    def move_selected_points_to_cluster(attr, old, new):
        global currently_selected_spike_indices
        if len(currently_selected_spike_indices) > 0 and new is not 'No cluster selected':
            remove_spikes_from_all_clusters(cluster_info_file, currently_selected_spike_indices)
            add_spikes_to_cluster(cluster_info_file, new, currently_selected_spike_indices)
            update_data_table()
            select_cluster_to_move_points_to.value = 'No cluster selected'
            user_info_edit.value = 'Selected clusters = ' + new

    select_cluster_to_move_points_to.on_change('value', move_selected_points_to_cluster)


    # undo selection button
    undo_selected_points_button = Button(label='Undo last selection')

    def on_button_undo_selection():
        global previously_selected_spike_indices
        tsne_source.selected['1d']['indices'] = previously_selected_spike_indices
        old = new = tsne_source.selected
        tsne_source.trigger('selected', old, new)

    undo_selected_points_button.on_click(on_button_undo_selection)

    # Sliders -------------------
    # use the fake data trick to call the callback only when the mouse is released (mouseup only works for CustomJS)

    # change visibility of non selected points Slider
    slider_non_selected_visibility = Slider(start=0, end=1, value=0.2, step=.02, callback_policy='mouseup',
                                            title='Alpha of not selected points',
                                            width=slider_size[0], height=slider_size[1])

    def on_slider_change_non_selected_visibility(attrname, old, new):
        global non_selected_points_alpha
        if len(source_fake_nsv.data['value']) > 0:
            non_selected_points_alpha = source_fake_nsv.data['value'][0]
            old = new = tsne_source.selected
            tsne_source.trigger('selected', old, new)

    source_fake_nsv = ColumnDataSource(data=dict(value=[]))
    source_fake_nsv.on_change('data', on_slider_change_non_selected_visibility)

    slider_non_selected_visibility.callback = CustomJS(args=dict(source=source_fake_nsv),
                                                       code="""
                                                            source.data = { value: [cb_obj.value] }
                                                            """)

    # change size of non selected points Slider
    slider_non_selected_size = Slider(start=0.5, end=10, value=2, step=0.5, callback_policy='mouseup',
                                      title='Size of not selected points',
                                      width=slider_size[0], height=slider_size[1])

    def on_slider_change_non_selected_size(attrname, old, new):
        global non_selected_points_size
        if len(source_fake_nss.data['value']) > 0:
            non_selected_points_size = source_fake_nss.data['value'][0]
            old = new = tsne_source.selected
            tsne_source.trigger('selected', old, new)

    source_fake_nss = ColumnDataSource(data=dict(value=[]))
    source_fake_nss.on_change('data', on_slider_change_non_selected_size)

    slider_non_selected_size.callback = CustomJS(args=dict(source=source_fake_nss),
                                                 code="""
                                                      source.data = { value: [cb_obj.value] }
                                                      """)

    # change size of selected points Slider
    slider_selected_size = Slider(start=0.5, end=10, value=2, step=0.5, callback_policy='mouseup',
                                  title='Size of selected points',
                                  width=slider_size[0], height=slider_size[1])

    def on_slider_change_selected_size(attrname, old, new):
        global selected_points_size
        if len(source_fake_ss.data['value']) > 0:
            selected_points_size = source_fake_ss.data['value'][0]
            old = new = tsne_source.selected
            tsne_source.trigger('selected', old, new)

    source_fake_ss = ColumnDataSource(data=dict(value=[]))
    source_fake_ss.on_change('data', on_slider_change_selected_size)

    slider_selected_size.callback = CustomJS(args=dict(source=source_fake_ss),
                                             code="""
                                                  source.data = { value: [cb_obj.value] }
                                                  """)

    # -------------------------------------------

    # Layout and session setup ------------------
    # align and make layout
    spike_figure.min_border_top = 50
    spike_figure.min_border_right = 10
    hist_figure.min_border_top = 50
    hist_figure.min_border_left = 10
    tsne_figure.min_border_right = 50

    if k4:
        lay = row(column(tsne_figure,
                         row(slider_non_selected_visibility, slider_non_selected_size, slider_selected_size),
                         row(spike_figure, hist_figure),
                         user_info_edit),
                 column(clusters_table,
                        button_show_clusters_of_selected_points,
                        button_merge_clusters_of_selected_points,
                        button_delete_cluster,
                        select_cluster_to_move_points_to,
                        new_cluster_name_edit,
                        button_show_all_clusters,
                        undo_selected_points_button,
                        heatmap_plot))
    else:
        lay = row(column(tsne_figure,
                         row(spike_figure, hist_figure)),
                  column(row(heatmap_plot, column(slider_non_selected_visibility,
                                                  slider_non_selected_size,
                                                  slider_selected_size)),
                         user_info_edit),
                  column(clusters_table,
                         button_show_clusters_of_selected_points,
                         button_merge_clusters_of_selected_points,
                         button_delete_cluster,
                         select_cluster_to_move_points_to,
                         new_cluster_name_edit,
                         button_show_all_clusters,
                         undo_selected_points_button))


    session = push_session(curdoc())
    session.show(lay)  # open the document in a browser
    session.loop_until_closed()  # run forever, requires stopping the interpreter in order to stop :)


