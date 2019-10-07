# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:07:35 2013

@author: George Dimitriadis
"""
import numpy as np
import scipy.signal as signal
import warnings
import BrainDataAnalysis.filters as filt
import math


def create_events(eventChannel, freq, threshold, minduration, pickOutOrIn):
    minduration = minduration * freq
    eventTransitions = np.diff(np.int32(eventChannel > threshold))
    eventSamples = np.nonzero(eventTransitions)
    eventSamples = np.squeeze(eventSamples)
    eventValues = eventTransitions[eventSamples]
    eventDurations = eventSamples[eventValues < 0] - eventSamples[eventValues > 0]
    validindices = np.squeeze(np.nonzero(eventDurations > minduration))
    if pickOutOrIn:
        eventSamples = eventSamples[eventValues > 0][validindices]
        eventValues = eventValues[eventValues > 0][validindices]
    else:
        eventSamples = eventSamples[eventValues < 0][validindices]
        eventValues = eventValues[eventValues < 0][validindices]
    return np.array([eventSamples, eventValues])


def create_piezosensor_events(event_channel, threshold, sampling_freq, filt_cutoff_freq, minduration, pickOutOrIn):
    if filt_cutoff_freq is not None:
        event_channel = filt.low_pass_filter(event_channel, sampling_freq, filt_cutoff_freq)
    event_channel = event_channel - np.mean(event_channel[0:10])
    event_transitions = np.diff(np.int32(event_channel < threshold))
    event_samples = np.nonzero(event_transitions)
    event_samples = np.squeeze(event_samples)
    event_values = event_transitions[event_samples]
    event_durations = event_samples[event_values < 0] - event_samples[event_values > 0]
    valid_indices = np.squeeze(np.nonzero(np.abs(event_durations) > minduration))
    if pickOutOrIn:
        event_samples = event_samples[event_values > 0][valid_indices]
        event_values = event_values[event_values > 0][valid_indices]
    else:
        event_samples = event_samples[event_values < 0][valid_indices]
        event_values = event_values[event_values < 0][valid_indices]
    return np.array([event_samples, event_values])


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
    spike_peaks = np.zeros(np.shape(spike_crossings))
    for i in range(len(spike_crossings)):
        if threshold > 0:
            offset = np.argmax(data_raw_spikes[int(spike_crossings[i]-(1e-3*sampling_freq)):
                                               int(spike_crossings[i]+(1e-3*sampling_freq))])
            peak = np.max(data_in_V[int(spike_crossings[i]-(1e-3*sampling_freq)):
                                          int(spike_crossings[i]+(1e-3*sampling_freq))])
        if threshold < 0:
            offset = np.argmin(data_raw_spikes[int(spike_crossings[i]-(1e-3*sampling_freq)):
                                               int(spike_crossings[i]+(1e-3*sampling_freq))])
            peak = np.min(data_in_V[int(spike_crossings[i]-(1e-3*sampling_freq)):
                                          int(spike_crossings[i]+(1e-3*sampling_freq))])
        spike_times[i] = spike_crossings[i] + offset - (1e-3*sampling_freq)
        spike_peaks[i] = peak
    return spike_times, spike_peaks, data_in_V


def spikedetect(data_raw_spikes, threshold_multiplier=4, single_max_threshold=False, inter_spike_time_distance=0.01,
                sampling_freq=32556, bad_channels=None):
    bad_channels = np.array(bad_channels)
    num_of_channels = np.shape(data_raw_spikes)[0]
    samples = np.arange(0, np.shape(data_raw_spikes)[1])
    thresholds = np.zeros(num_of_channels)
    for i in np.arange(0, num_of_channels):
        thresholds[i] = threshold_multiplier * np.median(np.abs(data_raw_spikes[i, :])) / .6745
        if not thresholds[i].any():
            print("Bad Channel " + str(i))
            if bad_channels is None:
                bad_channels = i
            else:
                bad_channels = np.append(bad_channels, i)

    if single_max_threshold:
        thresholds[:] = np.max(thresholds)

    inter_spike_distance = inter_spike_time_distance * sampling_freq

    all_channels_spike_times = np.empty(0)
    for i in np.arange(0, num_of_channels):
        if not is_element_in_array(bad_channels, i):
            print("Threshold = " + str(thresholds[i]))
            spike_times = np.array([x for x in samples if (data_raw_spikes[i, x] > thresholds[i])])
            diff_spikes_times = np.diff(spike_times)
            spike_times = np.array(
                [x for i, x in enumerate(spike_times[:-1]) if (diff_spikes_times[i] > inter_spike_distance)])
            print("spike_time size = " + str(np.size(spike_times)))
            if all_channels_spike_times.size is 0:
                all_channels_spike_times = np.append(all_channels_spike_times, spike_times)
                print("1 all_channels_spike_times size = " + str(np.size(all_channels_spike_times)))
            else:
                spike_times = np.array(
                    [x for i, x in enumerate(spike_times) if not is_element_in_array(all_channels_spike_times, x)])
                all_channels_spike_times = np.append(all_channels_spike_times, spike_times)
                print("2 all_channels_spike_times size = " + str(np.size(all_channels_spike_times)))
            print(i)
        print(i)

    return all_channels_spike_times


def is_element_in_array(array, element):
    return np.size(np.nonzero(array - element)) < np.size(array)


def remove_events(events, eventsToRemove):
    return np.delete(events, eventsToRemove, 1)


def create_solenoid_events(events, pickOutOrIn):
    if pickOutOrIn:
        finalEvents = events[:, events[1] > 0]
    else:
        finalEvents = events[:, events[1] < 0]
    return finalEvents


def time_lock_raw_data(data, events, times_to_cut, sampling_freq, baseline_time=None, sub_sample_freq=None,
                       high_pass_cutoff=None, rectify=False, low_pass_cutoff=None, avg_reref=False, keep_trials=False):
    """
    Time locks, baselines, high or low passes, and sub samples the data (in that order)
    """
    if np.ndim(events) == 2:
        events = events[0, :]
    number_of_trials = np.size(events, np.ndim(events) - 1)
    times_to_cut = np.array(times_to_cut)
    samples_to_cut = (times_to_cut * sampling_freq).astype(int)
    if np.size(np.shape(data)) > 1:
        number_of_channels = np.shape(data)[0]
    else:
        number_of_channels = 1
    number_of_samples = samples_to_cut[1] - samples_to_cut[0]
    time_axis = np.arange(times_to_cut[0], times_to_cut[1], (times_to_cut[1] - times_to_cut[0]) / number_of_samples)

    if sub_sample_freq:
        if keep_trials:
            if np.size(np.shape(data)) > 1:
                tl_singletrial_data = np.zeros(
                    [number_of_channels, math.ceil(number_of_samples * (sub_sample_freq / sampling_freq)),
                     number_of_trials])
            else:
                tl_singletrial_data = np.zeros(
                    [math.ceil(number_of_samples * (sub_sample_freq / sampling_freq)), number_of_trials])
        tl_avgtrials_data = np.empty(
            [number_of_channels, math.ceil(number_of_samples * (sub_sample_freq / sampling_freq))])
    else:
        if keep_trials:
            if np.size(np.shape(data)) > 1:
                tl_singletrial_data = np.zeros([number_of_channels, number_of_samples, number_of_trials])
            else:
                tl_singletrial_data = np.zeros([number_of_samples, number_of_trials])
        tl_avgtrials_data = np.zeros([number_of_channels, number_of_samples])

    for index in range(number_of_trials):
        temp_samples_to_cut = samples_to_cut + events[index]
        breakpoint = False

        # if there aren't enough points at the begining of the data set then disregard the event and move to the next one
        while np.min(temp_samples_to_cut) < 0:
            if temp_samples_to_cut[0] < 0:
                index = index + 1
                temp_samples_to_cut = samples_to_cut + events[index]
            elif temp_samples_to_cut[1] < 0:
                breakpoint = True
        if breakpoint:
            break
        if np.size(np.shape(data)) > 1:
            temp_data = data[:, int(temp_samples_to_cut[0]): int(temp_samples_to_cut[1])]
        else:
            temp_data = data[int(temp_samples_to_cut[0]): int(temp_samples_to_cut[1])]

        if avg_reref:  # rereference with mean over all channels
            temp_data = temp_data - np.mean(temp_data, 0)

        if high_pass_cutoff:
            temp_data = filt.high_pass_filter(temp_data, sampling_freq, high_pass_cutoff)
        elif low_pass_cutoff:
            temp_data = filt.low_pass_filter(temp_data, sampling_freq, low_pass_cutoff)
        if rectify:
            temp_data = np.abs(temp_data)
            if low_pass_cutoff:
                temp_data = filt.low_pass_filter(temp_data, sampling_freq, low_pass_cutoff)
            else:
                temp_data = filt.low_pass_filter(temp_data, sampling_freq, high_pass_cutoff / 2)
        elif not rectify and high_pass_cutoff:
            temp_data = filt.low_pass_filter(temp_data, sampling_freq, high_pass_cutoff / 2)

        if sub_sample_freq:
            temp_data, sub_time_axis = subsample_data(temp_data, time_axis, sampling_freq, sub_sample_freq,
                                                      filterType='fir', filterOrder=30)

        if baseline_time is not None:
            if sub_sample_freq:
                temp_data = baseline_correct(temp_data, sub_sample_freq, time_axis, baseline_time[0], baseline_time[1])
            else:
                temp_data = baseline_correct(temp_data, sampling_freq, time_axis, baseline_time[0], baseline_time[1])

        if keep_trials:
            if np.size(np.shape(data)) > 1:
                tl_singletrial_data[:, :, index] = temp_data
            else:
                tl_singletrial_data[:, index] = temp_data

        if index == 0:
            tl_avgtrials_data = temp_data
        elif index > 0:
            tl_avgtrials_data += temp_data

        if index % 10 == 0:
            print(index)

    tl_avgtrials_data /= number_of_trials
    if sub_sample_freq:
        time_axis = sub_time_axis

    returned_tuple = [tl_avgtrials_data, time_axis]
    if keep_trials:
        returned_tuple = [tl_singletrial_data, tl_avgtrials_data, time_axis]

    return returned_tuple


def subsample_basis_data(data, oldFreq, newFreq, filterType='Default', filterOrder=None):
    if int((oldFreq / newFreq) % int(oldFreq / newFreq)) is not 0:
        raise ArithmeticError("Subsampling can be done only with integer factors of Old Frequency / New Frequency")
    if filterType == 'Default':
        subData = signal.decimate(data, int(oldFreq / newFreq))
    else:
        if filterOrder == None:
            raise ArithmeticError("A filter order is required if the filter is to be specified")
        subData = signal.decimate(data, int(oldFreq / newFreq), filterOrder, filterType)
    return subData


def subsample_data(data, timeAxis, oldFreq, newFreq, filterType='Default', filterOrder=None):
    if filterType == 'Default':
        subData = subsample_basis_data(data, oldFreq, newFreq)
    else:
        subData = subsample_basis_data(data, oldFreq, newFreq, filterType, filterOrder)
    subTimeAxis = subsample_basis_data(timeAxis, oldFreq, newFreq, 'fir', 0)
    if (np.size(np.shape(data)) > 1 and np.size(subData, 1) < np.size(subTimeAxis, 0)) or (
            np.size(np.shape(data)) == 1 and np.size(subData) < np.size(subTimeAxis, 0)):
        warnings.warn("The length of the time is longer than the length of the data by " + str(
            np.size(subTimeAxis, 0) - np.size(subData, 1)) + " data point(s). These will be delete from the time axis.",
                      UserWarning)
        subTimeAxis = subTimeAxis[0:np.size(subData, 1)]
    return [subData, subTimeAxis]


def baseline_correct_basis(data, beginSample=0, endSample=-1):
    if np.size(np.shape(data)) > 1:
        baseline = np.mean(data[:, beginSample:endSample], 1)
    else:
        baseline = np.mean(data[beginSample:endSample])
    return np.transpose(np.transpose(data) - np.transpose(baseline))


def baseline_correct(data, Fsampling, timeAxis, beginTime=None, endTime=None):
    if beginTime == None:
        beginSample = 0
        startingTimeDif = 0
    else:
        startingTimeDif = beginTime - timeAxis[0]
        beginSample = math.ceil(startingTimeDif * Fsampling)
    if endTime == None:
        endSample = -1
        endingTimeDif = 0
    else:
        endingTimeDif = endTime - timeAxis[0]
        endSample = math.ceil(endingTimeDif * Fsampling)

    if startingTimeDif < 0:
        raise ArithmeticError("The starting time to plot must be after the starting time of the trial")
    if endingTimeDif < 0:
        raise ArithmeticError("The end time to plot must be after the starting time of the trial")
    return baseline_correct_basis(data, beginSample, endSample)


def split_list_pairwise(l,p):
    groups = []
    prev = None
    group = None
    for x in l:
        if prev is None or p(x,prev):
            group = []
            groups.append(group)
        group.append(x)
        prev = x
    return groups


def find_peaks(data_raw_spikes, threshold, inter_spike_time_distance=30, amp_gain=100,
                                  sampling_freq=30000, amp_y_digitization=65536, amp_y_range=10):

    scaling_factor = amp_y_range / (amp_y_digitization * amp_gain)
    data_in_v = (data_raw_spikes - np.mean(data_raw_spikes)) * scaling_factor

    derivative = np.diff(np.sign(np.diff(data_in_v)))
    if threshold > 0:
        derivative = derivative < 0
    else:
        derivative = derivative > 0

    peaks = derivative.nonzero()[0] + 1  # local max
    if threshold > 0:
        peaks = peaks[data_in_v[peaks] > threshold]
    else:
        peaks = peaks[data_in_v[peaks] < threshold]

    if inter_spike_time_distance > 0:
        gpeaks = split_list_pairwise(peaks, lambda x, p: x - p > inter_spike_time_distance)
        peaks = np.array([g[np.argmax([data_in_v[i] for i in g])] for g in gpeaks])
    return peaks, data_in_v

