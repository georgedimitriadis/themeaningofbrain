
import numpy as np
from numba import cuda
import accelerate.cuda.blas as cublas
import accelerate.cuda.sorting as sorting
import numba
from numba import float32, guvectorize, float64
import math
from timeit import default_timer as timer


# ----------------------------------------------------------------------------------------------------------------------
# Functions relating to calculating the high dimensional space distances, sorting them and picking the 3 x perplexity
#  closer ones

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

    threads_per_block = 512
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


def _sort_distances_get_knns(num_of_neighbours, distances_on_gpu, start_index, verbose=False):

    m = distances_on_gpu.shape[0]
    n = distances_on_gpu.shape[1]

    s = timer()

    radix_sort = sorting.RadixSort(maxcount=n, dtype=np.float32)
    selected_sorted_distances = np.empty((m, num_of_neighbours))
    selected_sorted_indices = np.empty((m, num_of_neighbours))
    for i in np.arange(m):
        values = np.array(np.arange(n) + start_index, dtype=np.float32)
        values_on_gpu = cuda.to_device(values)
        radix_sort.sort(keys=distances_on_gpu[i], vals=values_on_gpu)
        selected_sorted_distances[i, :] = distances_on_gpu[i, :num_of_neighbours].copy_to_host()
        selected_sorted_indices[i, :] = values_on_gpu[:num_of_neighbours].copy_to_host()

    e = timer()
    sort_time = e - s
    if verbose:
        print("Sorting Time:", "%.3f" % sort_time, "s")

    return selected_sorted_indices, selected_sorted_distances


def _segment_sort_distances_get_knns(num_of_neighbours, distances_on_gpu, start_index, number_of_sorts=100,
                                     verbose=False):

    m = distances_on_gpu.shape[0]
    n = distances_on_gpu.shape[1]
    k = num_of_neighbours + 1

    selected_sorted_distances = np.empty((m, num_of_neighbours))
    selected_sorted_indices = np.empty((m, num_of_neighbours))

    s = timer()
    p = np.append(np.arange(0, m, int(m / number_of_sorts)), m)
    for i in np.arange(1, p.shape[0]):
        delta_m = p[i] - p[i - 1]
        keys = np.ascontiguousarray(distances_on_gpu.copy_to_host()[p[i - 1]:p[i], :].reshape(n * delta_m))
        values = np.ascontiguousarray(np.tile(np.arange(n) + start_index, (delta_m, 1)).reshape(n * delta_m))
        segments = np.ascontiguousarray(np.arange(0, n * delta_m, n)[1:] + start_index)
        sorting.segmented_sort(keys=keys, vals=values, segments=segments)
        keys = np.reshape(keys, (delta_m, n))[:, :k]
        values = np.reshape(values, (delta_m, n))[:, :k]
        selected_sorted_distances[p[i - 1]:p[i], :] = keys[:, 1:]  # throw away the dist to self
        selected_sorted_indices[p[i - 1]:p[i], :] = values[:, 1:]
        if verbose:
            print('     Sorted ' + str(i) + ' of ' + str(p.shape[0] - 1) + ' segments of this iteration')
    e = timer()
    sort_time = e - s
    print("SORTING TIME:", "%.3f" % sort_time, "s")

    return selected_sorted_indices, selected_sorted_distances


def calculate_knn_distances_close_on_probe(template_features_sorted, indices_of_first_and_second_matrices,
                                           perplexity=100, verbose=True):

    start = timer()

    indices_of_first_matrices = indices_of_first_and_second_matrices[0]
    indices_of_second_matrices = indices_of_first_and_second_matrices[1]

    num_of_neighbours = perplexity * 3
    closest_indices = np.empty((template_features_sorted.shape[0], num_of_neighbours))
    closest_distances = np.empty((template_features_sorted.shape[0], num_of_neighbours))
    for matrix_index in np.arange(indices_of_first_matrices.shape[0]):
        first_matrix = np.array(template_features_sorted[indices_of_first_matrices[matrix_index][0]:
                                                         indices_of_first_matrices[matrix_index][1], :],
                                dtype=np.float32)
        second_matrix = np.array(template_features_sorted[indices_of_second_matrices[matrix_index][0]:
                                                          indices_of_second_matrices[matrix_index][1], :],
                                 dtype=np.float32)

        m = first_matrix.shape[0]
        n = second_matrix.shape[0]

        s = timer()
        if matrix_index != 0:
            del distances_on_gpu
        cuda.current_context().trashing.clear()

        if verbose:
            print('LOADING UP THE GPU')
        temp = np.array(np.zeros((m, n), dtype=np.float32))
        distances_on_gpu = cuda.to_device(np.asfortranarray(temp))
        e = timer()
        load_time = e - s
        if verbose:
            print("Loading matrix time:", "%.3f" % load_time, "s")

        if verbose:
            print('ITERATION NUMBER: ' + str(matrix_index + 1))

        _calculate_distances_on_gpu(a=first_matrix, b=second_matrix, distances_on_gpu=distances_on_gpu, verbose=verbose)

        gpu_mem = cuda.current_context().get_memory_info()
        available_gpu_mem = 0.9*gpu_mem[0]
        number_of_sorts = int(np.ceil((16 * n * m) / available_gpu_mem))  # 4 is the bytes per float32, 2 is the two
        # arrays that need to be loaded to gpu, the other factor of 2 is probably a doubling overhead in the algorithm

        if verbose:
            print('     Number of sorting segments = ' + str(number_of_sorts + 1))

        temp_indices, temp_distances = \
            _segment_sort_distances_get_knns(num_of_neighbours=num_of_neighbours, distances_on_gpu=distances_on_gpu,
                                             start_index=indices_of_second_matrices[matrix_index][0],
                                             number_of_sorts=number_of_sorts, verbose=verbose)

        closest_indices[indices_of_first_matrices[matrix_index][0]: indices_of_first_matrices[matrix_index][1],
                                :] = np.ascontiguousarray(temp_indices)
        closest_distances[indices_of_first_matrices[matrix_index][0]: indices_of_first_matrices[matrix_index][1],
                                  :] = np.ascontiguousarray(temp_distances)
        if verbose:
            print('FINISHED CALCULATING ' + str(matrix_index + 1) + ' OF ' + str(indices_of_first_matrices.shape[0]) +
                  ' ITERATIONS')

        end = timer()
        full_time = end - start
        if verbose:
            print("Spend Time:", "%.3f" % full_time, "s")

    return closest_indices, closest_distances

# ----------------------------------------------------------------------------------------------------------------------
# Function relating to computing the gradient of the force field in the low dimensional space
threadsperblock = (32, 32)
shape = 300

@cuda.jit(fastmath=True)
def _compute_sum_of_q_on_gpu(t_sne, partial_sum_q):

    i, j = cuda.grid(2)

    n = t_sne.shape[0]
    m = t_sne.shape[0]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # make and fill up with the q value shared memory among threads of one block
    partial_sm_sum_q = cuda.shared.array(threadsperblock, dtype=float32)

    partial_sm_sum_q[tx, ty] = 0
    if j >= 0 and j <= n - 1 and i >= 0 and i <= m - 1:
        # get the distance between 2 data points
        distance = 0
        for dim in range(t_sne.shape[1]):
            distance += (t_sne[i, dim] - t_sne[j, dim])*(t_sne[i, dim] - t_sne[j, dim])
        #distance = math.sqrt(distance)
        partial_sm_sum_q[tx, ty] = 1 / (1 + distance)

    cuda.syncthreads()

    # sum reduce the values of the shared memory array to generate a partial summation matrix (that needs to be summed
    # up further on the cpu)

    t = threadsperblock[0] // 2
    while t > 0:
        if tx < t:
            partial_sm_sum_q[tx, ty] = partial_sm_sum_q[tx, ty] + partial_sm_sum_q[tx + t, ty]
        t //= 2
        cuda.syncthreads()

    t = threadsperblock[0] // 2
    while t > 0:
        if ty < t and tx == 0:
            partial_sm_sum_q[tx, ty] = partial_sm_sum_q[tx, ty] + partial_sm_sum_q[tx, ty + t]
        t //= 2
        cuda.syncthreads()

    if tx == 0 and ty == 0:
        partial_sum_q[bx, by] = partial_sm_sum_q[0, 0]

    cuda.syncthreads()


@cuda.jit(fastmath=True)
def _compute_sum_of_q_on_gpu_sm(t_sne, partial_sum_q):
    i, j = cuda.grid(2)

    n = t_sne.shape[0]
    m = t_sne.shape[0]

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # make and fill up with the q value shared memory among threads of one block
    partial_sm_sum_q = cuda.shared.array(threadsperblock, dtype=float32)
    other_tsnes = cuda.shared.array(shape=(1024, 2), dtype=float32)
    this_tsne = cuda.shared.array(shape=2, dtype=float32)
    if tx >= 0 and tx < threadsperblock[0] and ty >= 0 and ty < threadsperblock[1]:
        if j >= 0 and j < n:
            other_tsnes[tx * ty + tx, :] = t_sne[j, :]

    cuda.syncthreads()

    if j == 0 and i >= 0 and i <= m - 1:
        this_tsne = t_sne[i, :]

    cuda.syncthreads()

    partial_sm_sum_q[tx, ty] = 0
    if tx >= 0 and tx < threadsperblock[0] and ty >= 0 and ty < threadsperblock[1]:
        if j >= 0 and j <= n - 1 and i >= 0 and i <= m - 1:

            # get the distance between 2 data points
            distance = 0
            for dim in range(t_sne.shape[1]):
                distance += (this_tsne[dim] - other_tsnes[tx * ty + tx, dim]) * (this_tsne[dim] - other_tsnes[tx * ty + tx, dim])
            # distance = math.sqrt(distance)
            partial_sm_sum_q[tx, ty] = 1 / (1 + distance)

    cuda.syncthreads()

    # sum up the values of the shared memory array to generate a partial summation matrix (that needs to be summed up
    # further on the cpu)

    t = threadsperblock[0] // 2
    while t > 0:
        if tx < t:
            partial_sm_sum_q[tx, ty] = partial_sm_sum_q[tx, ty] + partial_sm_sum_q[tx + t, ty]
        t //= 2
        cuda.syncthreads()

    t = threadsperblock[0] // 2
    while t > 0:
        if ty < t and tx == 0:
            partial_sm_sum_q[tx, ty] = partial_sm_sum_q[tx, ty] + partial_sm_sum_q[tx, ty + t]
        t //= 2
        cuda.syncthreads()

    if tx == 0 and ty == 0:
        partial_sum_q[bx, by] = partial_sm_sum_q[0, 0]

    cuda.syncthreads()

@cuda.jit(fastmath=True)
def _compute_repulsive_forces(t_sne, sum_q, delta):
    n, m = cuda.grid(2)

    if n >= 0 and n < t_sne.shape[0] and m >= 0 and m < t_sne.shape[0]:
        distance = 0
        for dim in range(t_sne.shape[1]):
            distance += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
        #distance = math.sqrt(distance)
        q = 1 / (1 + distance)

        mult = (q / sum_q[0]) * q

        if n != m:
            for dim in range(t_sne.shape[1]):
                delta[n, dim] -= (t_sne[n, dim] - t_sne[m, dim]) * mult

@cuda.jit(fastmath=True)
def _compute_attractive_forces(t_sne, values_p, indices_p, delta):
    n, m = cuda.grid(2)

    if n>= 0 and n < t_sne.shape[0] and m >= 0 and m < values_p.shape[1]:
        second_index = int(indices_p[n, m])
        distance = 0
        for dim in range(t_sne.shape[1]):
            distance += (t_sne[n, dim] - t_sne[second_index, dim]) * (t_sne[n, dim] - t_sne[second_index, dim])
        # distance = math.sqrt(distance)
        q = 1 / (1 + distance)

        value_p = values_p[n, m]

        mult = value_p / q

        if n != second_index:
            for dim in range(t_sne.shape[1]):
                delta[n, dim] += (t_sne[n, dim] - t_sne[second_index, dim]) * mult


@cuda.jit
def _compute_gradient_on_gpu_old(t_sne, values_p, indices_p, sum_q, delta):
    n, m = cuda.grid(2)

    if n >= 0 and n < t_sne.shape[0] and m >= 0 and m < t_sne.shape[0]:
        temp = 0
        for dim in range(t_sne.shape[1]):
            temp += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
        distance = math.sqrt(temp)
        q = 1 / (1 + distance)

        value_p = 0
        for i in range(indices_p.shape[1]):
            if indices_p[n, i] == m:
                value_p = values_p[n, i]
                break

        mult = (value_p - q / sum_q[0]) * q

        if n is not m:
            for dim in range(t_sne.shape[1]):
                delta[n, dim] += (t_sne[n, dim] - t_sne[m, dim]) * mult


@cuda.jit(fastmath=True)
def _compute_gradient_on_gpu(t_sne, values_p, indices_p, sum_q, delta):

    n, m, l = cuda.grid(3)

    if n >= 0 and n < t_sne.shape[0] and m >= 0 and m < t_sne.shape[0]:

        if l > 0 and l < indices_p.shape[1] + 1 and indices_p[n, l - 1] == m:
                p_value = values_p[n, l - 1]

                distance = 0
                for dim in range(t_sne.shape[1]):
                    distance += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
                #distance = math.sqrt(temp)
                q = 1 / (1 + distance)

                mult = (p_value - q / sum_q[0]) * q

                if n != m:
                    for dim in range(t_sne.shape[1]):
                        delta[n, dim] += (t_sne[n, dim] - t_sne[m, dim]) * mult

                return

        if l == 0:
            p_value = 0

            distance = 0
            for dim in range(t_sne.shape[1]):
                distance += (t_sne[n, dim] - t_sne[m, dim]) * (t_sne[n, dim] - t_sne[m, dim])
            #distance = math.sqrt(temp)
            q = 1 / (1 + distance)

            mult = (p_value - q / sum_q[0]) * q

            if n != m:
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


def _put_array_to_device(array, array_name, dtype=np.float64, verbose=True):
    s = timer()
    temp = np.array(array, dtype=dtype)
    d_array = cuda.to_device(temp)
    e = timer()
    if verbose:
        print('     Load ' + array_name + ' to device time: ' + str(e - s))
    return d_array


def compute_gradient_on_gpu(t_sne, indices_p, values_p, verbose=True):
    n = t_sne.shape[0]
    num_of_dims = t_sne.shape[1]
    delta = np.zeros((n, num_of_dims))

    s = timer()

    d_t_sne = _put_array_to_device(t_sne, 't_sne', dtype=np.float32, verbose=False)
    d_indices_p = _put_array_to_device(indices_p, 'indices_p', dtype=np.float32, verbose=False)
    d_values_p = _put_array_to_device(values_p, 'values_p', dtype=np.float32, verbose=False)
    d_delta = _put_array_to_device(delta, 'delta', dtype=np.float32, verbose=False)

    threads_per_block = threadsperblock
    blocks_per_grid_x = math.ceil(t_sne.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(t_sne.shape[0] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    partial_sum_q = np.empty(blocks_per_grid)
    d_partial_sum_q = _put_array_to_device(partial_sum_q, 'partial_sum_q', dtype=np.float32, verbose=False)

    _compute_sum_of_q_on_gpu[blocks_per_grid, threads_per_block](d_t_sne, d_partial_sum_q)
    partial_sum_q = d_partial_sum_q.copy_to_host()
    sum_q = np.sum(partial_sum_q)
    d_sum_q = _put_array_to_device(sum_q, 'sum_q',  dtype=np.float32, verbose=False)

    _compute_repulsive_forces[blocks_per_grid, threads_per_block](d_t_sne, d_sum_q, d_delta)

    cuda.synchronize()

    blocks_per_grid_y = math.ceil(values_p.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    _compute_attractive_forces[blocks_per_grid, threadsperblock](d_t_sne, d_values_p, d_indices_p, d_delta)
    delta = d_delta.copy_to_host()

    e = timer()
    if verbose:
        print('COMPUTING GRADIENT TIME: ' + str(e - s))

    return delta
