
import numpy as np
from numba import cuda
import accelerate.cuda.blas as cublas
import accelerate.cuda.sorting as sorting
import numba
from numba import float32, guvectorize
import math
from timeit import default_timer as timer


@guvectorize([(float32[:, :], float32[:])], '(m,k)->(m)', nopython=True)
def _create_dot_product(a, dots_a):
    for i in np.arange(a.shape[0]):
        dots_a[i] = np.dot(a[i, :], a[i, :])


@cuda.jit
def _sums_of_dots_gpu(dots_a, dots_b, s_o_dots):
    a_index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    if a_index < dots_a.shape[0]:
        for b_index in range(dots_b.shape[0]):
            s_o_dots[a_index, b_index] = dots_a[a_index] + dots_b[b_index]


def _calculate_distances_on_gpu(a, b, distances_on_gpu, verbose=False):
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    # calculate the inner product of each row for both matrices
    s1 = timer()
    dots_a = _create_dot_product(a)
    dots_b = _create_dot_product(b)
    e1 = timer()
    dot_time = e1 - s1
    if verbose:
        print("Making the dot products Time:", "%.3f" % dot_time, "s")

    # calculate the ||a||^2 + ||b||^2 sum of dot products matrix (a.a + b.b) on the gpu and save on the gpu_temp matrix
    s2 = timer()
    ddots_a = cuda.to_device(np.asfortranarray(dots_a))
    ddots_b = cuda.to_device(np.asfortranarray(dots_b))

    threads_per_block = 32
    blocks_per_grid = math.ceil(ddots_a.shape[0] / threads_per_block)
    _sums_of_dots_gpu[blocks_per_grid, threads_per_block](ddots_a, ddots_b, distances_on_gpu)
    numba.cuda.synchronize()

    e2 = timer()
    s_o_dot_time = e2 - s2
    if verbose:
        print("Summing the dot products on the GPU Time:", "%.3f" % s_o_dot_time, "s")

    # calculate the -2<a,b> cross dot products matrix and do the sum (||a||^2 + ||b||^2) -2<a,b>
    s3 = timer()
    da = cuda.to_device(np.asfortranarray(a))
    db = cuda.to_device(np.asfortranarray(b))
    blas = cublas.Blas()
    blas.gemm('N', 'T', m, n, k, -2.0, da, db, 1.0, distances_on_gpu)
    numba.cuda.synchronize()

    e3 = timer()
    time_gemm = e3 - s3
    if verbose:
        print("cuBLAS gemm Time:", "%.3f" % time_gemm, "s")


def _segment_sort_transposed_distances_get_knns(num_of_neighbours, distances_on_gpu, number_of_sorts,
                                                verbose=False):

    m = distances_on_gpu.shape[0]  # all spikes
    n = distances_on_gpu.shape[1]  # part of spikes in iteration

    selected_sorted_distances = np.empty((n, num_of_neighbours))
    selected_sorted_indices = np.empty((n, num_of_neighbours))

    s = timer()
    p = np.append(np.arange(0, n, int(n / number_of_sorts)), n)
    for i in np.arange(1, p.shape[0]):
        delta_n = p[i] - p[i - 1]
        keys = np.ascontiguousarray(distances_on_gpu.copy_to_host()[:, p[i - 1]:p[i]].transpose().reshape(m * delta_n))
        values = np.ascontiguousarray(np.tile(np.arange(m), (delta_n, 1)).reshape(m * delta_n))
        segments = np.ascontiguousarray(np.arange(m, m * delta_n, m))
        sorting.segmented_sort(keys=keys, vals=values, segments=segments)
        keys = np.reshape(keys, (delta_n, m))[:, :num_of_neighbours]
        values = np.reshape(values, (delta_n, m))[:, :num_of_neighbours]
        selected_sorted_distances[p[i - 1]:p[i], :] = keys[:, :]
        selected_sorted_indices[p[i - 1]:p[i], :] = values[:, :]
        if verbose:
            print('     Sorted ' + str(i) + ' of ' + str(p.shape[0] - 1) + ' segments of this iteration')
    e = timer()
    sort_time = e - s
    print("SORTING TIME:", "%.3f" % sort_time, "s")

    return selected_sorted_indices, selected_sorted_distances


def calculate_knn_distances(template_features_sparse_clean, perplexity=100, mem_usage=0.9, verbose=True):
    start = timer()
    num_of_neighbours = perplexity * 3 + 1
    m = template_features_sparse_clean.shape[0]

    closest_indices = np.empty((m, num_of_neighbours))
    closest_distances = np.empty((m, num_of_neighbours))

    gpu_mem = cuda.current_context().get_memory_info()
    available_gpu_mem = 0.5 * gpu_mem[0]

    n = int(np.ceil(available_gpu_mem / (4 * m)))

    number_of_iters = int(np.ceil(m / n))

    indices_of_second_matrices = [(i, i + n) for i in np.arange(0, number_of_iters * (n - 1), n)]
    indices_of_second_matrices[-1] = (indices_of_second_matrices[-1][0], m)

    first_matrix = np.array(template_features_sparse_clean, dtype=np.float32)

    for iter in np.arange(number_of_iters):
        second_matrix = np.array(template_features_sparse_clean[indices_of_second_matrices[iter][0]:
                                                                indices_of_second_matrices[iter][1], :],
                                 dtype=np.float32)

        s = timer()
        if iter != 0:
            del distances_on_gpu
        cuda.current_context().deallocations.clear()  # for numba version 0.30
        # cuda.current_context().trashing.clear()  # for numba version 0.25

        if verbose:
            print('LOADING UP THE GPU')
        temp = np.array(np.zeros((m, second_matrix.shape[0]), dtype=np.float32))
        distances_on_gpu = cuda.to_device(np.asfortranarray(temp))
        e = timer()
        load_time = e - s
        if verbose:
            print("Loading matrix time:", "%.3f" % load_time, "s")

        if verbose:
            print('ITERATION NUMBER: ' + str(iter + 1))

        _calculate_distances_on_gpu(a=first_matrix, b=second_matrix, distances_on_gpu=distances_on_gpu, verbose=verbose)

        gpu_mem = cuda.current_context().get_memory_info()
        available_gpu_mem = 0.5 * gpu_mem[0]
        number_of_sorts = int(np.ceil((16 * n * m) / available_gpu_mem))  # 4 is the bytes per float32, 2 is the two
        # arrays that need to be loaded to gpu, the other factor of 2 is probably a doubling overhead in the algorithm


        if verbose:
            print('     Number of sorting segments = ' + str(number_of_sorts + 1))

        temp_indices, temp_distances = \
            _segment_sort_transposed_distances_get_knns(num_of_neighbours=num_of_neighbours,
                                                        distances_on_gpu=distances_on_gpu,
                                                        number_of_sorts=number_of_sorts, verbose=verbose)

        closest_indices[indices_of_second_matrices[iter][0]: indices_of_second_matrices[iter][1], :] = \
            np.ascontiguousarray(temp_indices)
        closest_distances[indices_of_second_matrices[iter][0]: indices_of_second_matrices[iter][1], :] = \
            np.ascontiguousarray(temp_distances)
        if verbose:
            print('FINISHED CALCULATING ' + str(iter + 1) + ' OF ' + str(number_of_iters) +
                  ' ITERATIONS')

        end = timer()
        full_time = end - start
        if verbose:
            print("Spend Time:", "%.3f" % full_time, "s")

    return closest_indices, np.sqrt(np.abs(closest_distances))

