__author__ = 'George Dimitriadis'




import numpy as np

from ..constants import FIFF
from ..meas_info import Info
from ..base import _BaseRaw
from ...utils import verbose, logger
from ...externals.six import string_types


class read_raw_from_memmap(_BaseRaw):
    """Raw object from from memmaped data (2D) on the HD (using np.memmap)

    Parameters
    ----------
    data_filename : a string = the filename to the 2D data on the HD.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure. Requires nchan to be set.
    dtype : the type of the data on the HD.
    order : str either 'chan_time' or 'time_chan'. Specifies if the channels
        are the 0th or 1st axis. The data in the output raw object is always chan_time
    verbose : mne's verbose level
    """
    @verbose
    def __init__(self, data_filename, info,  dtype=np.uint16, order='chan_time', verbose=None):
        fdata = np.memmap(data_filename, dtype)

        numchannels = info['nchan']
        numsamples = int(len(fdata) / numchannels)

        if order == 'chan_time':
             data = fdata.reshape(numchannels, numsamples)
        elif order == 'time_chan':
            data = fdata.reshape(numsamples, numchannels)
            data = data.T

        logger.info('Creating raw object with memmaped data of %s type, %s number of channels and %s number of time points'
                    % (dtype.__name__, data.shape[0], data.shape[1]))


        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        self.verbose = verbose
        self.cals = cals
        self.rawdir = None
        self.proj = None
        self.comp = None
        self._filenames = list()
        self.preload = True
        self.info = info
        self._data = data
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp,
                                self.last_samp + 1) / info['sfreq']
        self._projectors = list()
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                    self.first_samp, self.last_samp,
                    float(self.first_samp) / info['sfreq'],
                    float(self.last_samp) / info['sfreq']))
        logger.info('Ready.')