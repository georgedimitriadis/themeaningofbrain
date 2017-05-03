import numpy as np
import os
import sys
from numba import cuda
from timeit import default_timer as timer
import scipy.spatial.distance as spdist

import accelerate.cuda.blas as cublas
import accelerate.cuda.sorting as sorting
import numba
from numba import float32, uint32, guvectorize, jit
import math



# define the path for all files and filename of raw data----------------------------------------------------------------
base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                  r'Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'


#  stuff to compute gaussian perplexity

# load (spikes x 3*per) matrices of the closest in hd space distances and their respective spike indices
selected_sorted_indices = np.load(os.path.join(base_folder, 'selected_hdspace_sorted_indices.npy'))
selected_sorted_distances = np.load(os.path.join(base_folder, 'selected_hdspace_sorted_distances.npy'))

perplexity = 100

k = selected_sorted_indices.shape[1]
n = selected_sorted_indices.shape[0]
dbl_min = sys.float_info[3]
dbl_max = sys.float_info[0]

ind_p = selected_sorted_indices.astype(np.int)
val_p = np.empty((n, k))
for spike in np.arange(100):
    beta = 1.0
    found = False
    min_beta = -dbl_max
    max_beta = dbl_max
    tolerance = 1e-5

    iter = 0
    sum_p = 0
    while not found and iter < 200:
        cur_distances = selected_sorted_distances[spike, :]
        cur_p = np.exp(-beta * cur_distances)
        sum_p = dbl_min + np.sum(cur_p)
        H = np.sum(beta * cur_distances * cur_p) / sum_p + np.log(sum_p)

        H_diff = H - np.log(perplexity)
        if H_diff < tolerance and -H_diff < tolerance:
            found = True
        else:
            if H_diff > 0:
                min_beta = beta
                if max_beta == dbl_max or max_beta == -dbl_max:
                    beta *= 2.0
                else:
                    beta = (beta + max_beta) / 2.0
            else:
                max_beta = beta
                if min_beta == -dbl_max or min_beta == dbl_max:
                    beta /= 2.0
                else:
                    beta = (beta + min_beta) / 2.0
        iter += 1

    cur_p /= sum_p

    val_p[spike, :] = cur_p












from os.path import join
from struct import calcsize, unpack

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


folder = r'E:\George\SourceCode\Repos\t_sne_bhcuda\bin\windows'
distances = np.empty((50, 30))
indices = np.empty((50, 30))

for i in np.arange(50):
    if i < 10:
        file_d = 'distances_row00000'+str(i)+'.dat'
        file_i = 'indices_row00000' + str(i) + '.dat'
    else:
        file_d = 'distances_row0000'+ str(i) +'.dat'
        file_i = 'indices_row0000' + str(i) + '.dat'
    with open(join(folder, file_d), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        results = [_read_unpack('{}i'.format(result_dims), output_file) for _ in range(result_samples)]
    distances[i, :] = np.array(results).squeeze()
    with open(join(folder, file_i), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        results = [_read_unpack('{}i'.format(result_dims), output_file) for _ in range(result_samples)]
    indices[i, :] = np.array(results).squeeze()
distances = np.sqrt(np.array(distances))


file = 'interim_000019.dat'
with open(join(folder, file), 'rb') as output_file:
    # The first two integers are the number of samples and the dimensionality
    result_samples, result_dims = _read_unpack('ii', output_file)
    y = [_read_unpack('{}d'.format(result_dims), output_file) for _ in range(result_samples)]
y = np.array(y)



from numba import jitclass, jit
from numba import int32, float32
import inspect

spec = [
    ('a', int32),
    ('b', float32[:])
]


class Test:
    def __init__(self, a, b=None):
        if b is None:
            print(a)

    def jitf(self, a):
        return testf(self, a)

@jit(nopython=True)
def testf(test, a):
    a += 1
    return a


T = Test(a=3)
t = T.jitf(a=10)


