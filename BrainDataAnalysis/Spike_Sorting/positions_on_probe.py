import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spikesorting_tsne import constants as ct
import pandas as pd


def _get_relevant_channels_with_threshold(threshold, template):
    amplitude = np.nanmax(template) - np.nanmin(template)
    points_over_threshold = np.argwhere(template > (np.nanmax(template) - threshold * amplitude))
    channels_over_threshold = np.unique(points_over_threshold[:, 1])
    return channels_over_threshold


def _get_relevant_channels_over_median_peaks(threshold, template):
    median = np.median(np.nanmin(template, axis=0))
    std = np.std(np.nanmin(template, axis=0))
    points_under_median = np.argwhere(template < (median - threshold*std))
    channels_over_threshold = np.unique(points_under_median[:, 1])
    return channels_over_threshold


def _normalize(L, normalizeFrom=0, normalizeTo=1):
    '''normalize values of a list to make its min = normalizeFrom and its max = normalizeTo'''
    vMax = max(L)
    vMin = min(L)
    return [(x-vMin)*(normalizeTo - normalizeFrom) / (vMax - vMin) for x in L]
# ------------------


def generate_probe_positions_of_spikes(base_folder, binary_data_filename, number_of_channels_in_binary_file,
                                       used_spikes_indices=None, position_mult=2.25, threshold=0.1):
    """
    Generate positions (x, y coordinates) for each spike on the probe. This function assumes that the spikes were
    generated with the kilosort algorithm so the base_folder holds all the necessary .npy arrays.
    In order for this function to find which channels are the most relevant in each spike it looks into the spike's
    assigned template (a channels x time points array in spike_templates.npy). It then find the minimum points of all
    channels, takes their median and their standard deviation and for each channel creates the difference between the
    minimum and the median. Finally it demarcates the relevant to the template channels by keeping the ones whose
    difference is larger than a number of times (threshold) over the standard deviation.
    It then picks the relevant channels of the spike's raw data, finds the differences between the minimum value
    and the channel's time series median value (over time), orders the channels according to these differences and
    assigns weights between 0 and 1 (0 for a difference of 0, 1 for a maximum difference).
    It finally finds the x, y positions of the selected channels and adds to the position of the largest difference
    channel the weighted average positions of the remaining selected channels

    :param base_folder: the folder name into which the kilosort result .npy arrays are
    :type base_folder: string
    :param binary_data_filename: the name of the binary file that holds the raw data that were originally passed to kilosort
    :type binary_data_filename: string
    :param number_of_channels_in_binary_file: How many channels does the binary file have (this is different to the number
    of channels that are set to active in kilosort)
    :type number_of_channels_in_binary_file: int
    :param used_spikes_indices: which of the spikes found by kilosort should be considered.
    :type used_spikes_indices: int[:]
    :param threshold: the number of times the standard deviation should be larger than the difference between a
    channel's minimum and the median of the minima of all channels in order to demarcate the channel as
    relevant to the spike

    :type threshold: float
    :return:  The position of each spike on the probe, the distance of eac h spike on the probe from the 0, 0 of the
    probe, the indices of the original ordering of the spikes on the new order sorted according to their distance on
    the probe, the distance of each spike on the probe sorted
     :rtype: float32[len(used_spike_indices), 2], float32[len(used_spike_indices)], int[len(used_spike_indices)], float32[len(used_spike_indices)]

    """
    # Load the required data from the kilosort folder
    channel_map = np.load(os.path.join(base_folder, 'channel_map.npy'))
    active_channel_map = np.squeeze(channel_map, axis=1)
    channel_positions = np.load(os.path.join(base_folder, 'channel_positions.npy'))

    spike_templates = np.load(os.path.join(base_folder, ct.SPIKE_TEMPLATES_FILENAME))
    templates = np.load(os.path.join(base_folder, ct.TEMPLATES_FILENAME))

    data_raw = np.memmap(os.path.join(base_folder, binary_data_filename),
                         dtype=np.int16, mode='r')

    number_of_timepoints_in_raw = int(data_raw.shape[0] / number_of_channels_in_binary_file)
    data_raw_kilosorted = np.reshape(data_raw, (number_of_channels_in_binary_file, number_of_timepoints_in_raw), order='F')

    spike_times = np.squeeze(np.load(os.path.join(base_folder, ct.SPIKE_TIMES_FILENAME)).astype(np.int))

    time_points = 50
    if used_spikes_indices is None:
        used_spikes_indices = np.arange(0, len(spike_times))

    # Run the loop over all spikes to get the positions
    counter = 0
    weighted_average_postions = np.empty((len(used_spikes_indices), 2))
    spike_distance_on_probe = np.empty(len(used_spikes_indices))
    for spike_index in np.arange(0, len(used_spikes_indices)):
        spike_raw_data = data_raw_kilosorted[active_channel_map,
                                             (spike_times[used_spikes_indices[spike_index]]-time_points):
                                             (spike_times[used_spikes_indices[spike_index]]+time_points)]
        template = templates[spike_templates[used_spikes_indices[spike_index]], :, :].squeeze()
        relevant_channels = _get_relevant_channels_over_median_peaks(threshold, template)

        spike_raw_data_median_over_time = np.median(spike_raw_data, axis=1)
        peaks_to_median = spike_raw_data_median_over_time - spike_raw_data.min(axis=1)
        peaks_to_median = peaks_to_median[relevant_channels]

        relevant_channels_sorted = [v for (k, v) in sorted(zip(peaks_to_median, relevant_channels), reverse=True)]

        peaks_to_median_sorted = sorted(peaks_to_median, reverse=True)
        peaks_to_median_sorted.append(np.median(spike_raw_data_median_over_time[relevant_channels]))

        weights = _normalize(peaks_to_median_sorted)[:-1]
        relevant_channels_positions = channel_positions[relevant_channels_sorted]

        pos_x = relevant_channels_positions[0, 0]
        pos_y = relevant_channels_positions[0, 1]

        new_pos_x = pos_x - np.mean(((pos_x - relevant_channels_positions[:, 0]) * weights)[1:])
        new_pos_y = pos_y - np.mean(((pos_y - relevant_channels_positions[:, 1]) * weights)[1:])
        weighted_average_postions[spike_index, :] = [new_pos_x, new_pos_y]
        spike_distance_on_probe[spike_index] = np.sqrt(np.power(new_pos_x, 2) + np.power(new_pos_y, 2))

        counter += 1
        if counter % 5000 == 0:
            print('Completed ' + str(counter) + ' spikes')
    weighted_average_postions = weighted_average_postions * position_mult

    # sort according to position on probe
    spike_indices_sorted_by_probe_distance = np.array([b[0] for b in sorted(enumerate(spike_distance_on_probe),
                                                                            key=lambda dist: dist[1])])
    spike_distances_on_probe_sorted = np.array([b[1] for b in sorted(enumerate(spike_distance_on_probe),
                                                                     key=lambda dist: dist[1])])

    np.save(os.path.join(base_folder, ct.WEIGHTED_SPIKE_POSITIONS_FILENAME), weighted_average_postions)

    return weighted_average_postions, spike_distance_on_probe, \
        spike_indices_sorted_by_probe_distance, spike_distances_on_probe_sorted


def generate_probe_positions_of_templates(base_folder, threshold=0.1, new_templates_array=None):
    """
    Generate positions (x, y coordinates) for each template found by kilosort on the probe or passed to it by the
    new_templates_array.
    This function assumes that the base_folder holds all the necessary .npy arrays.

    If no new_templates_array is passed it will look for the templates.npy file (created by kilosort) which is the
    average of all spikes for each template (so a (templates x time x channels) data cube). It will also try to find the
    file template_marking.npy which is produced after cleaning using the spikesort_tsne_guis.clean_kilosort_templates
    GUI. If this is found only the non noise templates will have their position evaluated. If not found all templates
    will be considered.

    If a new_templates_array is passed (a data cube of either (templates x time x channels) or (templates x channels x time)
    dimensions) then this will be used to calculate the positions.

    In order for this function to find which channels are the most relevant in each template it looks into the
    template (a (channels x time) array). It then find the minimum points of all
    channels, takes their median and their standard deviation and for each channel creates the difference between the
    minimum and the median. Finally it demarcates the relevant to the template channels by keeping the ones whose
    difference is larger than a number of times (threshold) over the standard deviation.
    It then picks the relevant channels of the spike's raw data, finds the differences between the minimum value
    and the channel's time series median value (over time), orders the channels according to these differences and
    assigns weights between 0 and 1 (0 for a difference of 0, 1 for a maximum difference).
    It finally finds the x, y positions of the selected channels and adds to the position of the largest difference
    channel the weighted average positions of the remaining selected channels

    :param base_folder: the folder name into which the kilosort result .npy arrays are
    :type base_folder: string
    :param threshold: the number of times the standard deviation should be larger than the difference between a
    channel's minimum and the median of the minima of all channels in order to demarcate the channel asvrelevant to the
    spike

    :type threshold: float
    :param new_templates_array: an array that is the average over spikes of all templates
    :type new_templates_array: float32[templates x channels x time]
    :return: weighted_average_postions : the positions of the templates on the probe
    :rtype: weighted_average_postions : float32[len(used_spike_indices) x 2]

    """
    # Load the required data from the kilosort folder
    channel_positions = np.load(os.path.join(base_folder, 'channel_positions.npy'))
    if new_templates_array is None:
        try:
            templates = np.load(os.path.join(base_folder, ct.TEMPLATES_FILENAME))
        except FileNotFoundError:
            exit('No new_templates_array passed and no templates.npy found in folder')
        try:
            template_markings = np.load(os.path.join(base_folder, ct.TEMPLATE_MARKING_FILENAME))
        except FileNotFoundError:
            template_markings = np.ones((len(templates)))
        templates = templates[template_markings > 0, :, :]

    else:
        if new_templates_array.shape[1] > new_templates_array.shape[2]:
            templates = np.reshape(new_templates_array, (new_templates_array.shape[0],
                                                         new_templates_array.shape[2],
                                                         new_templates_array.shape[1]))
        else:
            templates = new_templates_array

    # Run the loop over all templates to get the positions
    counter = 0
    templates_positions = []
    for template in templates:
        relevant_channels = _get_relevant_channels_over_median_peaks(threshold, template)

        template_median_over_time = np.median(template, axis=0)
        peaks_to_median = template_median_over_time - template.min(axis=0)
        peaks_to_median = peaks_to_median[relevant_channels]

        relevant_channels_sorted = [v for (k, v) in sorted(zip(peaks_to_median, relevant_channels), reverse=True)]

        peaks_to_median_sorted = sorted(peaks_to_median, reverse=True)
        peaks_to_median_sorted.append(np.median(template_median_over_time[relevant_channels]))

        weights = _normalize(peaks_to_median_sorted)[:-1]
        relevant_channels_positions = channel_positions[relevant_channels_sorted]

        pos_x = relevant_channels_positions[0, 0]
        pos_y = relevant_channels_positions[0, 1]

        new_pos_x = pos_x - np.mean(((pos_x - relevant_channels_positions[:, 0]) * weights)[1:])
        new_pos_y = pos_y - np.mean(((pos_y - relevant_channels_positions[:, 1]) * weights)[1:])
        templates_positions.append([new_pos_x, new_pos_y])
        counter += 1
        if not (counter % 100):
            print('Completed ' + str(counter) + ' templates')

    templates_positions = np.array(templates_positions)

    np.save(os.path.join(base_folder, ct.WEIGHTED_TEMPLATE_POSITIONS_FILENAME), templates_positions)

    return np.array(templates_positions)


def get_y_spread_regions_of_bad_channel_groups(base_folder, bad_channel_groups):

    channel_positions = np.load(os.path.join(base_folder, 'channel_positions.npy'))
    bad_channel_groups_y_spreads = []
    for bc_group in bad_channel_groups:
        bc_positions = channel_positions[bc_group]
        top = bc_positions[:, 1].max()
        bottom = bc_positions[:, 1].min()
        bad_channel_groups_y_spreads.append([bottom, top])

    return bad_channel_groups_y_spreads


def view_spike_positions(spike_positions, brain_regions, probe_dimensions, labels_offset=80, font_size=20):
    """
    Plot the spike positions as a scatter plot on a probe marked with brain regions

    :param spike_positions: the x,y positions of the spikes
    :type spike_positions: (np.array((N,2)))
    :param brain_regions: a dictionary with keys the names of the brain regions underneath the demarcating lines and
    values the y position on the probe of the demarcating lines

    :type brain_regions: dict
    :param probe_dimensions: the x and y limits of the probe
    :type probe_dimensions: (np.array(2))

    """

    fig = plt.figure()
    ax = fig.add_axes([0.08, 0.05, 0.9, 0.9])
    ax.scatter(spike_positions[:, 0], spike_positions[:, 1], s=5)
    ax.set_xlim(0, probe_dimensions[0])
    ax.set_ylim(0, probe_dimensions[1])
    ax.yaxis.set_ticks(np.arange(0, probe_dimensions[1], 100))
    ax.tick_params(axis='y', direction='in', length=5, width=1, colors='b')
    for region in brain_regions:
        ax.text(2, brain_regions[region] - labels_offset, region, fontsize=font_size)
        ax.plot([0, probe_dimensions[0]], [brain_regions[region], brain_regions[region]], 'k--', linewidth=2)
    return fig, ax


def view_grouped_templates_positions(base_folder, brain_regions, probe_dimensions, position_multiplier=1,
                                     bad_channel_regions=None, template_info=None, labels_offset=80,
                                     font_size=20, dot_sizes=None,
                                     func_to_run_on_click=None, args_of_func=None):
    """

    :param base_folder: the folder where all the npy arrays (template_markings etc.) are saved
    :type base_folder: string
    :param brain_regions: a dictionary with keys the names of the brain regions underneath the demarcating lines and
    values the y position on the probe of the demarcating lines
    :param probe_dimensions: the dimensions of the probe
    :type probe_dimensions: np.array(2)
    :type brain_regions: (dict{string: float})
    :param position_multiplier: a number multiplying the positions so that the numbers are not the arbitrary ones from
    the prb file but correspond to the length of the probe

    :type position_multiplier: float
    :param template_info: If provided the template_info will be used to define the types of the templates. It assumes
    the length of the template_info and of the loaded weighted_template_positions array is the same unless the
    template_info has template positions (position X and position Y column) in it. In this case these are used. Also
    the template_info is used to know which template is clicked on the figure for the on_pick event

    :type template_info: pd.Dataframe
    :param labels_offset: offset of the labels on the plot
    :type labels_offset: (int)
    :param font_size: the font size of the labels
    :type font_size: (int)
    :param func_to_run_on_click: The function to run on a click of a scatter point. It assumes that the first argument
    it needs is the template row of the template_info that was click.
    :type func_to_run_on_click: Func
    :param args_of_func: The arguments of the function to run on on_pick (after the template itself)
    :type args_of_func: list of objects
    :return:
    """

    template_positions = np.squeeze(
        position_multiplier * np.load(os.path.join(base_folder, ct.TEMPLATE_POSITIONS_FILENAME)))

    if template_info is None:
        template_markings = np.load(os.path.join(base_folder, ct.TEMPLATE_MARKING_FILENAME))
        clean_template_markings = np.squeeze(template_markings[np.argwhere(template_markings > 0)])
    else:
        clean_template_markings = np.empty((len(template_info)))
        for t in ct.types:
            clean_template_markings[template_info['type'] == ct.types[t]] = t
        if ~np.isnan(template_info.iloc[0]['position X']):
            template_positions = template_info[['position X', 'position Y']].values * position_multiplier

    def on_pick(event):
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        ind = event.ind[0]
        x = template_positions[ind, 0]
        y = template_positions[ind, 1]
        print('________________________')
        print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
        print('Position: {}, {}'.format(str(x), str(y)))
        print('------------------------')
        if template_info is not None:
            template_number = template_info.iloc[ind]['template number']
            print('Template number = {}'.format(template_number))
            print('Firing frequency = {}'.format(template_info.iloc[ind]['firing rate']))
            print('Number of spikes = {}'.format(template_info.iloc[ind]['number of spikes']))
            print('________________________')
            if func_to_run_on_click is not None:
                if args_of_func is None:
                    func_to_run_on_click(template_info.iloc[ind])
                else:
                    func_to_run_on_click(template_info.iloc[ind], *args_of_func)

    types = np.flipud(np.unique(clean_template_markings))
    fig = plt.figure()
    ax = fig.add_axes([0.08, 0.05, 0.9, 0.9])
    fig.canvas.callbacks.connect('pick_event', on_pick)

    tolerance = 1
    cm = plt.cm.cool

    type_to_color = {1: (0, 61/255, 1, 1), 2: (27/255, 221/255, 206/255, 1), 3: (99/255, 214/255, 39/255, 1),
                     4: (255/255, 183/255, 0/255, 1), 5: (100/255, 100/255, 100/255, 1),
                     6: (170 / 255, 170 / 255, 170 / 255, 1), 7: (240/255, 240/255, 240/255, 1)}
    type_to_size = {1: 60, 2: 50, 3: 40, 4: 40, 5: 40, 6: 40, 7: 40}

    colors = np.array(len(template_positions) * [(0, 0, 0, 1)]).astype(float)
    sizes = np.array(len(template_positions) * [40.0])

    for type in types:
        indices_of_templates_of_type = np.squeeze(np.argwhere(clean_template_markings == type)).astype(np.int)
        if np.size(indices_of_templates_of_type) < 2:
            colors[indices_of_templates_of_type] = type_to_color[type]
        else:
            colors[indices_of_templates_of_type] = [type_to_color[type]]
        if dot_sizes is None:
            sizes[indices_of_templates_of_type] = type_to_size[type]

    if dot_sizes is not None:
        sizes = dot_sizes

    ax.scatter(template_positions[:, 0], template_positions[:, 1], s=sizes, c=colors,
               picker=tolerance)

    ax.set_xlim(0, probe_dimensions[0])
    ax.set_ylim(0, probe_dimensions[1])
    ax.yaxis.set_ticks(np.arange(0, probe_dimensions[1], 100))
    ax.tick_params(axis='y', direction='in', length=5, width=1, colors='b')

    if bad_channel_regions is not None:
        for bc_region in bad_channel_regions:
            bc_region = np.array(bc_region) * position_multiplier
            ax.add_patch(Rectangle((0, bc_region[0]),
                                   100, bc_region[1] - bc_region[0],
                                   facecolor="grey", alpha=0.5))

    for region in brain_regions:
        ax.text(2, brain_regions[region] - labels_offset, region, fontsize=font_size)
        ax.plot([0, probe_dimensions[0]], [brain_regions[region], brain_regions[region]], 'k--', linewidth=2)

    return fig, ax
