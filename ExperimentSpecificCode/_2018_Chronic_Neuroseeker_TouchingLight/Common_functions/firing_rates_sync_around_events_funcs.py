
import numpy as np
import matplotlib.pyplot as plt
from ExperimentSpecificCode._2018_Chronic_Neuroseeker_TouchingLight.Common_functions import \
    events_sync_funcs as sync_funcs

VIDEO_FRAME_RATE = 120
SAMPLING_FREQUENCY = 20000


def get_avg_firing_rates_around_events(spike_rates, event_time_points, ev_video_df, time_around_event=5):

    frames_of_events = sync_funcs.time_point_to_frame_from_video_df(ev_video_df, event_time_points)
    frames_around_event = time_around_event * VIDEO_FRAME_RATE

    firing_rate_around_events = np.zeros((len(event_time_points), len(spike_rates), 2 * frames_around_event))

    for f in np.arange(len(frames_of_events)):
        frame = frames_of_events[f]
        firing_rate_around_events[f, :, :] = spike_rates[:, frame - frames_around_event:
                                                            frame + frames_around_event]

    avg_firing_rate_around_events= firing_rate_around_events.mean(axis=0)

    return avg_firing_rate_around_events


def get_neurons_following_pattern_around_an_event(avg_firing_rate_around_event, time_around_pattern,
                                                  pattern_regions_to_compare=[0, 0.66, 1.3, 2],
                                                  comparison_factor=3, comparison_direction='increase', baserate=0.1):

    frames_around_pattern = time_around_pattern * VIDEO_FRAME_RATE
    first_start = int(frames_around_pattern * pattern_regions_to_compare[0])
    first_end = int(frames_around_pattern * pattern_regions_to_compare[1])
    second_start = int(frames_around_pattern * pattern_regions_to_compare[2])
    second_end = int(frames_around_pattern * pattern_regions_to_compare[3])

    firing_rates_following_pattern = []
    index_of_neurons_following_pattern = []

    for n in np.arange(len(avg_firing_rate_around_event)):
        neuron = avg_firing_rate_around_event[n]

        if comparison_direction == 'increase':

            if neuron[first_start:first_end].mean() > baserate:
                if neuron[first_start:first_end].mean() * comparison_factor < \
                        neuron[second_start:second_end].mean():

                    firing_rates_following_pattern.append(neuron)
                    index_of_neurons_following_pattern.append(n)

        elif comparison_direction == 'decrease':

            if neuron[first_start:first_end].mean() > baserate:
                if neuron[first_start:first_end].mean() > \
                        neuron[second_start:second_end].mean() * comparison_factor:

                    firing_rates_following_pattern.append(neuron)
                    index_of_neurons_following_pattern.append(n)

    firing_rates_following_pattern = np.array(firing_rates_following_pattern)
    index_of_neurons_following_pattern = np.array(index_of_neurons_following_pattern)

    print(firing_rates_following_pattern.shape)

    return index_of_neurons_following_pattern, firing_rates_following_pattern


def show_firing_rates_around_event(firing_rates):
    fig = plt.figure(0)
    fig.set_figheight(2)
    fig.set_figwidth(10)
    ax = fig.add_subplot(111)
    ax.vlines(x=0, ymin=0, ymax=len(firing_rates))
    frames_around_event = int(firing_rates.shape[1] /2)
    ax.imshow(firing_rates, vmax=firing_rates.max(), vmin=0,
              extent=[-frames_around_event * 8.3,
                      frames_around_event * 8.3,
                      0,
                      len(firing_rates)])
    ax.set_aspect(200)


def show_rasters(index, firing_rates_neuron_index, firing_rates, template_info, spike_info,
                 event_times, frames_around_event, fig1, fig2):
    neuron_index = firing_rates_neuron_index[index]
    firing_rate = firing_rates[neuron_index]

    fig1.clear()
    ax1 = fig1.add_subplot(111)
    ax1.plot(firing_rate)
    ax1.vlines(x=frames_around_event, ymin=firing_rate.min(),
               ymax=firing_rate.max())

    largest_decrease_neurons_spikes = template_info.iloc[neuron_index]['spikes in template']
    largest_decrease_neurons_spike_times = spike_info.loc[np.isin(spike_info['original_index'],
                                                                  largest_decrease_neurons_spikes)]['times'].values.\
        astype(np.int64)

    neuron_raster = []
    for trial in event_times:
        neuron_raster.append((largest_decrease_neurons_spike_times - trial) / SAMPLING_FREQUENCY)

    neuron_raster = np.array(neuron_raster)

    fig2.clear()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(neuron_raster, np.tile(np.arange(len(neuron_raster)),
                                                        neuron_raster.shape[1]),
                s=10)
    time_points_around_event = frames_around_event * VIDEO_FRAME_RATE
    ax2.set_xlim(-time_points_around_event / SAMPLING_FREQUENCY,
                 time_points_around_event / SAMPLING_FREQUENCY)

    return None