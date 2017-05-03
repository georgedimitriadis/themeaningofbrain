

import numpy as np
import math
from numba import cuda
from numba import float64
import sys
from timeit import default_timer as timer
import os


threadsperblock = (32, 32)

@cuda.jit
def _compute_sum_of_q_on_gpu(t_sne, partial_sum_q):

    i, j = cuda.grid(2)

    n = t_sne.shape[0]
    m = t_sne.shape[0]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # make and fill up with the q value shared memory among threads of one block
    block_shared_mem = cuda.shared.array(threadsperblock, dtype=float64)

    block_shared_mem[tx, ty] = 0
    if j >= 0 and j <= n - 1 and i >= 0 and i <= m - 1:
        # get the distance between 2 data points
        temp = 0
        for dim in range(t_sne.shape[1]):
            temp += (t_sne[i, dim] - t_sne[j, dim])*(t_sne[i, dim] - t_sne[j, dim])
        #distance = math.sqrt(temp)
        block_shared_mem[tx, ty] = 1 / (1 + temp)

    cuda.syncthreads()

    # sum up the values of the shared memory array to generate a partial summation matrix (that needs to be summed up
    # further on the cpu)

    t = threadsperblock[0] // 2
    while t > 0:
        if tx < t:
            block_shared_mem[tx, ty] = block_shared_mem[tx, ty] + block_shared_mem[tx + t, ty]
        t //= 2
        cuda.syncthreads()

    t = threadsperblock[0] // 2
    while t > 0:
        if ty < t and tx == 0:
            block_shared_mem[tx, ty] = block_shared_mem[tx, ty] + block_shared_mem[tx, ty + t]
        t //= 2
        cuda.syncthreads()

    if tx == 0 and ty == 0:
        partial_sum_q[bx, by] = block_shared_mem[0, 0]

    cuda.syncthreads()


@cuda.jit(fastmath=True)
def _compute_gradient_on_gpu(t_sne, values_p, indices_p, sum_q, delta):

    n, m, l = cuda.grid(3)

    if n >= 0 and n < t_sne.shape[0] and m >= 0 and m < t_sne.shape[0]:

        if l > 0 and l < indices_p.shape[1] + 1 and indices_p[n, l - 1] == m:
                p_value = values_p[n, l - 1]

                temp = 0
                for dim in range(t_sne.shape[1]):
                    temp += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
                #distance = math.sqrt(temp)
                q = 1 / (1 + temp)

                mult = (p_value - q / sum_q[0]) * q

                if n is not m:
                    for dim in range(t_sne.shape[1]):
                        delta[n, dim] += (t_sne[n, dim] - t_sne[m, dim]) * mult

                return

        if l == 0:
            p_value = 0

            temp = 0
            for dim in range(t_sne.shape[1]):
                temp += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
            #distance = math.sqrt(temp)
            q = 1 / (1 + temp)

            mult = (p_value - q / sum_q[0]) * q

            if n is not m:
                for dim in range(t_sne.shape[1]):
                    delta[n, dim] += (t_sne[n, dim] - t_sne[m, dim]) * mult

            return


@cuda.jit(fastmath=True)
def _compute_iteration_on_gpu(t_sne, values_p, indices_p, sum_q, delta, uy, gains, momentum, eta):

    n, m, l = cuda.grid(3)

    # Make sure we are within range for m and n
    if n >= 0 and n < t_sne.shape[0] and m >= 0 and m < t_sne.shape[0]:

        # Calculate the delta. Use l as an index to the 2nd dimension of the p values and indices (3 * perplexity)
        if l > 0 and l < indices_p.shape[1] + 1 and indices_p[n, l - 1] == m:
                p_value = values_p[n, l - 1]

                distance = 0
                for dim in range(t_sne.shape[1]):
                    distance += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
                q = 1 / (1 + distance)

                mult = (p_value - q / sum_q[0]) * q

                if n is not m:
                    for l in range(t_sne.shape[1]):
                        delta[n, l] += (t_sne[n, l] - t_sne[m, l]) * mult

        if l == 0:
            p_value = 0

            distance = 0
            for dim in range(t_sne.shape[1]):
                distance += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
            q = 1 / (1 + distance)

            mult = (p_value - q / sum_q[0]) * q

            if n is not m:
                for dim in range(t_sne.shape[1]):
                    delta[n, dim] += (t_sne[n, dim] - t_sne[m, dim]) * mult

        cuda.syncthreads()

        # Calculate the new t-sne. Use l as an index to the dimensionality of the t-sne space (2 or 3)
        if m == 0 and l >= 0 and l < t_sne.shape[1]:
            sign_check = delta[n, l] * uy[n, l]
            if sign_check >= 0:
                gains[n, l] *= 0.95
            else:
                gains[n, l] += 0.05
            if gains[n, l] < 0.01:
                gains[n, l] = 0.01

            uy[n, l] = momentum[0] * uy[n, l] - eta[0] * gains[n, l] * delta[n, l]

            t_sne[n, l] += uy[n, l]

            cuda.syncthreads()

'''
gains[np.argwhere(np.sign(dy) != np.sign(uy))] += 0.05
gains[np.argwhere(np.sign(dy) == np.sign(uy))] *= 0.95
gains[np.argwhere(gains < 0.01)] = 0.01

# update gradient
uy = momentum * uy - eta * gains * dy
y += uy

# zero mean solution
y = pylab.demean(y, axis=0)
'''
def _put_array_to_device(array, array_name, dtype=np.float64, verbose=True):
    s = timer()
    temp = np.array(array, dtype=dtype)
    d_array = cuda.to_device(temp)
    e = timer()
    if verbose:
        print('     Load ' + array_name + ' to device time: ' + str(e - s))
    return d_array


def main():
    base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'

    perplexity = 100


    indices_p = np.load(os.path.join(base_folder, r'indices_p.npy'))
    values_p = np.load(os.path.join(base_folder, r'values_p.npy'))
    num_dims = 2

    extender = 1
    n = indices_p.shape[0] * extender
    print(n)
    tsne = np.array(np.random.random((n, num_dims)), dtype=np.float32)

    indices_p = np.tile(indices_p, (extender, 1))
    values_p = np.tile(values_p, (extender, 1))

    verbose = True
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(tsne.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(tsne.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    num_of_dims = tsne.shape[1]
    partial_sum_q = np.zeros(blockspergrid)
    cuda.profile_start()
    t1s = timer()
    d_tsne = _put_array_to_device(tsne, 't_sne', np.float64, verbose)
    d_partial_sum_q = _put_array_to_device(partial_sum_q, 'partial_sum_q', np.float64, verbose)
    t2s = timer()
    _compute_sum_of_q_on_gpu[blockspergrid, threadsperblock](d_tsne, d_partial_sum_q)
    partial_sum_q = d_partial_sum_q.copy_to_host()
    t2e = timer()
    print('Time to run the sum of q on the gpu = ' + str(t2e - t2s))

    t3s = timer()
    sum_q = np.sum(partial_sum_q)
    t3e = timer()
    print('Time to run the sum of q on the cpu = ' + str(t3e - t3s))
    print(sum_q)

    d_sum_q = _put_array_to_device(sum_q, 'sum_q', np.float64, verbose)


    delta = np.zeros((n, num_of_dims))
    uy = np.zeros((n, num_dims))
    gains = np.ones((n, num_dims))
    momentum = [0.5]
    eta = [200]
    d_indices_p = _put_array_to_device(indices_p, 'indices_p', dtype=np.float64, verbose=verbose)
    d_values_p = _put_array_to_device(values_p, 'values_p', dtype=np.float64, verbose=verbose)
    d_delta = _put_array_to_device(delta, 'delta', dtype=np.float64, verbose=verbose)
    d_uy = _put_array_to_device(uy, 'uy', dtype=np.float64, verbose=verbose)
    d_gains = _put_array_to_device(gains, 'gains', dtype=np.float64, verbose=verbose)
    d_momentum = _put_array_to_device(momentum, 'momentum', dtype=np.float64, verbose=False)
    d_eta = _put_array_to_device(eta, 'eta', dtype=np.float64, verbose=False)

    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(tsne.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(tsne.shape[0] / threadsperblock[1])
    blockspergrid_z = math.ceil((indices_p.shape[1] + 1) / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)


    t4s = timer()
    _compute_iteration_on_gpu[blockspergrid, threadsperblock](d_tsne, d_values_p, d_indices_p, d_sum_q, d_delta, d_uy,
                                                              d_gains, d_momentum, d_eta)
    #_compute_gradient_on_gpu[blockspergrid, threadsperblock](d_tsne, d_values_p, d_indices_p, d_sum_q, d_delta)
    t4e = timer()
    print('Time to run one iteration on gpu = ' + str(t4e - t4s))
    t_sne = d_tsne.copy_to_host()
    t1e = timer()
    cuda.profile_stop()
    print('Time to copy t_sne to host = ' + str(t1e-t4e))
    print('Total time = ' + str(t1e - t1s))




if __name__ == "__main__":
    main()