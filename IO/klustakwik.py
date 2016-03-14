import numpy as np


# For klusta the data must be in a flat array with Time1Chan1, Time1Chan2,... Time1ChanN, Time2Chan1,... structure
def make_dat_file(raw_data, filename, num_channels, time_limits=None):
    if not time_limits:
        time_limits = [0, raw_data.shape[1]]
    raw_data_ivm_klusta = np.reshape(np.transpose(raw_data[:, time_limits[0]:time_limits[1]]),
                                     (num_channels * (time_limits[1] - time_limits[0])))
    raw_data_ivm_klusta = (raw_data_ivm_klusta - 20000).astype('int16') #The -20K is to bring the number within the int16 range

    raw_klusta_size = len(raw_data_ivm_klusta)

    temp = np.memmap(filename,
                     dtype=np.int16,
                     mode='w+',
                     shape=raw_klusta_size)
    temp[:] = raw_data_ivm_klusta
    del temp