__author__ = 'George Dimitriadis'


import numpy as np
import scipy.signal as signal
from BrainDataAnalysis._Old_Structures import Constants as ct


def low_pass_filter(data, Fsampling, Fcutoff, filterType='but', filterOrder=None, filterDirection='twopass'):
    """
    Low passes the data at the Fcutoff frequency.
    filterType = ´but´ (butterworth) (default) OR ´fir´
    filterOrder = the order of the filter. For the default butterworth filter it is 6
    filterDirection = FilterDirection which defines whether the filter is passed over the data once (and how) or twice
    """
    Wn = np.float32(Fcutoff / (Fsampling / 2.0))
    if filterType == 'fir':
        if filterOrder == None:
            raise ArithmeticError("A filter order is required if the filter is to be a fir and not a but")
        (b, a) = signal.firwin(filterOrder + 1, Wn, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    else:
        if filterOrder == None:
            filterOrder = 6
        (b, a) = signal.butter(filterOrder, Wn, btype='lowpass', analog=0, output='ba')

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1

    if filterDirection == ct.FilterDirection.TWO_PASS:
        filteredData = signal.filtfilt(b, a, data, axis)
    elif filterDirection == ct.FilterDirection.ONE_PASS:
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
    elif filterDirection == ct.FilterDirection.ONE_PASS_REVERSE:
        data = np.fliplr(data)
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
        filteredData = np.fliplr(filteredData)
    return filteredData


def high_pass_filter(data, Fsampling, Fcutoff, filterType='but', filterOrder=None, filterDirection='twopass'):
    """
    High passes the data at the Fcutoff frequency.
    filterType = ´but´ (butterworth) (default) OR ´fir´
    filterOrder = the order of the filter. For the default butterworth filter it is 6
    filterDirection = FilterDirection which defines whether the filter is passed over the data once (and how) or twice
    """
    Wn = np.float32(Fcutoff / (Fsampling / 2.0))
    if filterType == 'fir':
        if filterOrder == None:
            raise ArithmeticError("A filter order is required if the filter is to be a fir and not a but")
        (b, a) = signal.firwin(filterOrder + 1, Wn, width=None, window='hamming', pass_zero=False, scale=True, nyq=1.0)
    else:
        if filterOrder == None:
            filterOrder = 6
        (b, a) = signal.butter(filterOrder, Wn, btype='highpass', analog=0, output='ba')

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1

    if filterDirection == ct.FilterDirection.TWO_PASS:
        filteredData = signal.filtfilt(b, a, data, axis)
    elif filterDirection == ct.FilterDirection.ONE_PASS:
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
    elif filterDirection == ct.FilterDirection.ONE_PASS_REVERSE:
        data = np.fliplr(data)
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
        filteredData = np.fliplr(filteredData)
    return filteredData


def band_iir_filter(data, Fsampling, stopband=[49,51], passband = [47,53], gpass = 1, gstop = 20, filterType='butter', filterDirection='twopass', pad_samples = None, pad_type = None, **pad_kwargs):
    """
    High passes the data at the Fcutoff frequency.
    stopband = the inside corner points in Hz
    passband = the outside corner points in Hz
    If the stopband encompasses the passband the filter is a passband filter, otherwise it is a stopband
    gpass = the minimum power (in possitive dB) of the pass band
    gstop = the maximum power (in possitive dB) of the stop band
    filterType = 'butter' (default), 'cheby1', 'cheby2', 'ellip', 'bessel'
    filterDirection = FilterDirection which defines whether the filter is passed over the data once (and how) or twice
    pad_samples = the number of samples to pad with at each side (if None and pad_type is not None then default pad_samples = np.shape(data)[-1])
    pad_type = the type of paddding (see numpy.pad's mode for more info)
    pad_kwargs = extra padding arguments (see numpy.pad for more info
    """
    ws = [np.float(x / (Fsampling / 2.0)) for x in stopband]
    wp = [np.float(x / (Fsampling / 2.0)) for x in passband]

    (b, a) = signal.iirdesign(ws = ws, wp = wp, gpass = gpass, gstop = gstop,  ftype  = filterType, output='ba')

    dims = data.ndim
    axis = 0
    if dims == 2:
        axis = 1

    if pad_type:
        data = np.pad(data, pad_samples, pad_type, pad_kwargs)

    if filterDirection == ct.FilterDirection.TWO_PASS:
        filteredData = signal.filtfilt(b, a, data, axis)
    elif filterDirection == ct.FilterDirection.ONE_PASS:
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
    elif filterDirection == ct.FilterDirection.ONE_PASS_REVERSE:
        data = np.fliplr(data)
        filteredData = signal.lfilter(b, a, data, axis, zi=None)
        filteredData = np.fliplr(filteredData)

    if pad_type:
        if dims == 1:
            filteredData = filteredData[pad_samples:-pad_samples]
        if dims == 2:
            filteredData = filteredData[:,pad_samples:-pad_samples]

    return filteredData


# def pad_data(data, pad_samples = None, type = 'zeros'):
#
#     if len(np.shape(data) >1):
#         two_d = true;
#         num_chans = np.shape(data)[0]
#         num_points = np.shape(data)[1]
#     else:
#         num_points = np.shape(data)[0]
#
#     if not pad_samples:
#         pad_samples = num_points
#
#     if type == 'zeros':
#         if two_d:
#             pad = np.zeros((num_chans , num_points + 2*pad_samples))
#         else:
#             pad = np.zeros((num_points))
#
#
#     return data
