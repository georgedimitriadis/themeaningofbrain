

import pyeemd
import numpy as np
import time
from joblib import Parallel, delayed
import sys

from os.path import join


def emd_in_blocks(signal, result_filename, result_dtype=np.int16, block_length=10000, emd_type='ceemdan',
                  num_imfs=15, ensemble_size=25, noise_strength=0.01, S_number=20, num_siftings=100):

    n_channels = signal.shape[0]
    n_timepoints = signal.shape[1]
    emd_shape = (n_channels, num_imfs, n_timepoints)  # channels X imfs X time points

    imfs = np.memmap(result_filename, dtype=result_dtype, mode='r+', shape=emd_shape)

    if n_timepoints < block_length:
        block_length = n_timepoints

    n_blocks = int(np.floor(n_timepoints / block_length))
    extra_points = int(0.1*block_length)
    cutoff = int(extra_points / 2)

    t0 = time.process_time()
    for b in np.arange(n_blocks):
        if b == 0:
            initial_offset = 0
            shift_cutoff = 0
        else:
            initial_offset = extra_points
            shift_cutoff = cutoff

        data = signal[0, b*block_length - initial_offset:(b+1)*block_length]

        temp_imfs = pyeemd.ceemdan(data, num_imfs=num_imfs, ensemble_size=ensemble_size,
                                                      noise_strength=noise_strength, S_number=S_number,
                                                      num_siftings=num_siftings)

        offset = initial_offset - shift_cutoff
        temp_imfs = temp_imfs[:, offset:].astype(result_dtype)

        imfs[0, :, b*block_length - offset:(b+1)*block_length] = temp_imfs
    print(time.process_time() - t0)


def emd_per_channel(channel, channel_data, result_filename, result_dtype, emd_shape,
                    num_imfs, ensemble_size, noise_strength, S_number, num_siftings):

    imfs = np.memmap(result_filename, dtype=result_dtype, mode='r+', shape=emd_shape)

    print('Channel {} imfs started calculating'.format(str(channel)))

    t0 = time.process_time()
    channel_imfs = pyeemd.ceemdan(channel_data, num_imfs=num_imfs, ensemble_size=ensemble_size,
                               noise_strength=noise_strength, S_number=S_number,
                               num_siftings=num_siftings)

    imfs[channel, :, :] = channel_imfs
    del channel_imfs
    del imfs
    print('Channel {} imfs finished after {} minutes'.format(str(channel), str((time.process_time() - t0)/60)))


def emd(signal, result_filename, result_dtype=np.int16,
        num_imfs=13, ensemble_size=25, noise_strength=0.01, S_number=20, num_siftings=100):


    n_channels = signal.shape[0]
    n_timepoints = signal.shape[1]
    emd_shape = (n_channels, num_imfs, n_timepoints)  # channels X imfs X time points

    #imfs = np.memmap(result_filename, dtype=result_dtype, mode='r+', shape=emd_shape)
    #del imfs

    Parallel(n_jobs=5)(delayed(emd_per_channel)(channel,
                                                signal[channel, :],
                                                result_filename,
                                                result_dtype,
                                                emd_shape,
                                                num_imfs,
                                                ensemble_size,
                                                noise_strength,
                                                S_number,
                                                num_siftings)
                        for channel in np.arange(64, n_channels))


def run_emd(args):

    downsampled_lfp_filename = args[1]
    result_filename = args[2]
    num_imfs = int(args[4])
    ensemble_size = int(args[5])
    noise_strength = float(args[6])
    S_number = int(args[7])
    num_siftings = int(args[8])

    t = args[3]
    if t == 'int8':
        result_dtype = np.int8
    elif t == 'uint8':
        result_dtype = np.uint8
    elif t == 'int16':
        result_dtype = np.int16
    elif t == 'uint16':
        result_dtype = np.uint16
    elif t == 'int32':
        result_dtype = np.int32
    elif t == 'uint32':
        result_dtype = np.uint32
    elif t == 'float' or t == 'float32':
        result_dtype = np.float32
    elif t == 'float64':
        result_dtype = np.float64

    downsampled_lfp = np.load(downsampled_lfp_filename)

    emd(downsampled_lfp, result_filename, result_dtype=result_dtype,
        num_imfs=num_imfs, ensemble_size=ensemble_size, noise_strength=noise_strength,
        S_number=S_number, num_siftings=num_siftings)


def load_memmaped_imfs(filename, num_of_imfs, num_of_channels, dtype=np.int16):

    imfs = np.memmap(filename, dtype=dtype, mode='r')

    num_of_timepoints = int(len(imfs) / (num_of_channels * num_of_imfs))

    imfs = np.reshape(imfs, (num_of_channels, num_of_imfs, num_of_timepoints))

    return imfs


if __name__ == "__main__":
    args = sys.argv
    run_emd(args)



