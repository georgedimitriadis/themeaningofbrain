
from os.path import dirname, isdir, join as path_join
from struct import calcsize, pack, unpack
from platform import system
import numpy as np


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def find_exe_file():
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
        if len(current_dir) == 3:
            return None
    tsne_path = path_join(dir_to_exe, exe_file)

    return tsne_path


def save_data_for_barneshut(files_dir, sorted_distances, sorted_indices, num_of_dims, perplexity, theta, eta,
                            exageration, iterations, random_seed, verbose):

    sorted_indices = np.array(sorted_indices, dtype=np.int32)
    num_of_spikes = len(sorted_distances)
    num_of_nns = len(sorted_distances[0])

    filename = 'data.dat'

    with open(path_join(files_dir, filename), 'wb') as data_file:
        # Write the t_sne_bhcuda header
        data_file.write(pack('dddiiiiiii', theta, eta, exageration, num_of_spikes, num_of_dims, num_of_nns, iterations,
                             random_seed, verbose, perplexity))
        # Write the data
        for sample in sorted_distances:
            data_file.write(pack('{}d'.format(len(sample)), *sample))

        for sample in sorted_indices:
            data_file.write(pack('{}i'.format(len(sample)), *sample))


def load_tsne_result(files_dir, filename='result.dat'):
    # Read and pass on the results
    with open(path_join(files_dir, filename), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file) for _ in range(result_samples)]

        return np.array(results)


def load_barneshut_data(files_dir, filename='data.dat', data_has_exageration=True):
    data_file = path_join(files_dir, filename)

    print('Loading previously calculated high dimensional distances')

    with open(data_file, 'rb') as output_file:
        if data_has_exageration:
            theta, eta, exageration, num_of_spikes, num_of_dims, num_of_nns, iterations, \
            random_seed, verbose, perplexity = _read_unpack('dddiiiiiii', output_file)

            parameters_dict = {'theta': theta, 'eta': eta, 'exageration': exageration,  'num_of_spikes': num_of_spikes,
                               'num_of_dims': num_of_dims, 'num_of_nns': num_of_nns, 'iterations': iterations,
                               'random_seed': random_seed, 'verbose': verbose, 'perplexity': perplexity}

        else:
            theta, eta, num_of_spikes, num_of_dims, num_of_nns, iterations, \
            random_seed, verbose, perplexity = _read_unpack('ddiiiiiii', output_file)

            parameters_dict = {'theta': theta, 'eta': eta, 'num_of_spikes': num_of_spikes,
                               'num_of_dims': num_of_dims, 'num_of_nns': num_of_nns, 'iterations': iterations,
                               'random_seed': random_seed, 'verbose': verbose, 'perplexity': perplexity}

        sorted_distances = np.array(
            [_read_unpack('{}d'.format(num_of_nns), output_file) for _ in range(num_of_spikes)])

        sorted_indices = np.array(
            [_read_unpack('{}i'.format(num_of_nns), output_file) for _ in range(num_of_spikes)])

    print('     Size of distances matrix: ' + str(sorted_distances.shape))
    print('     Size of indices matrix: ' + str(sorted_indices.shape))

    return sorted_distances, sorted_indices, parameters_dict

