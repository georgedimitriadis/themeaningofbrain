__author__ = 'George Dimitriadis'

import numpy as np
import BrainDataAnalysis._Old_Structures.Structures as dm
import BrainDataAnalysis._Old_Structures.Constants as ct


voltage_step_size = 0.195e-6

def load_raw_data(filename, numchannels=32, dtype=np.uint16):
    fdata = np.memmap(filename, dtype)
    numsamples = int(len(fdata) / numchannels)
    dataMatrix = np.reshape(fdata, (numsamples, numchannels))
    dataMatrix = dataMatrix.T
    dimensionOrder = dm.DataDimensions.fromIndividualDimensions(ct.Dimension.CHANNELS, ct.Dimension.TIME)
    data = dm.Data(dataMatrix, dimensionOrder)
    return data


def load_raw_event_trace(filename, number_of_channels=8, channel_used=None, dtype=np.int32):
    fdata = np.fromfile(filename, dtype)
    numsamples = int(len(fdata) / number_of_channels)
    dataMatrix = np.reshape(fdata, (numsamples, number_of_channels))
    dataMatrix = dataMatrix.T
    if number_of_channels == 1 and channel_used is not None:
        raise ArithmeticError("If you set only one channel do not set its number")
    if number_of_channels > 1 and channel_used is None:
        raise ArithmeticError("If you set more than one channels set the number where the events are")
    if number_of_channels == 1:
        dataStream = dataMatrix[0, :]
    else:
        dataStream = dataMatrix[channel_used, :]
    dimensionOrder = dm.DataDimensions.fromIndividualDimensions(ct.Dimension.TIME)
    data = dm.Data(dataStream, dimensionOrder)
    return data
