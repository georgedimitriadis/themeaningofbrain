
from os import replace
from os.path import dirname, isdir, join as path_join
from struct import calcsize, pack, unpack
from platform import system
import sys
import numpy as np


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def _find_exe_dir():
    exe_dir = 'bin'
    exe_file = 'Barnes_Hut'
    if system() == 'Windows':
        exe_dir = 'Scripts'
        exe_file = 'Barnes_Hut.exe'

    dir_to_exe = None
    current_dir = dirname(__file__)
    while dir_to_exe == None:
        if isdir(path_join(current_dir, exe_dir)):
            dir_to_exe = path_join(current_dir, exe_dir)
        current_dir = dirname(current_dir)

    tsne_path = path_join(dir_to_exe, exe_file)

    return tsne_path


def save_data_for_tsne(files_dir, y, col_p, val_p, theta, perplexity, eta, iterations, verbose):

    n = len(y)
    no_dims = len(y[0])
    k = len(col_p[0])

    filename = 'data.dat'

    with open(path_join(files_dir, filename), 'wb') as data_file:
        # Write the t_sne_bhcuda header
        data_file.write(pack('ddiiiiii', theta, eta, n, no_dims, k, iterations, verbose, perplexity))
        # Write the data
        for sample in y:
            data_file.write(pack('{}d'.format(len(sample)), *sample))

        for sample in col_p.flatten():
            data_file.write(pack('I', sample))

        for sample in val_p.flatten():
            data_file.write(pack('d', sample))



def load_tsne_result(files_dir):
    filename = 'result.dat'
    # Read and pass on the results
    with open(path_join(files_dir, filename), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        print(result_samples)
        print(result_dims)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file) for _ in range(result_samples)]

        return results
