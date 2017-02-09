
# TESTING WHICH METHOD FOR CALCULATING EUCLIDEAN DISTANCE IS FASTER
import numpy as np
from numba import cuda
from timeit import default_timer as timer
import scipy.spatial.distance as spdist
import os

import accelerate.cuda.blas as cublas
import accelerate.cuda.sorting as sorting
import numba
from numba import float32, uint32, guvectorize, jit
import math
from TSne_Numba import spikes, gpu



# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- TEST LOW DIM SPACE DISTANCES CALCULATIONS -----------------------------------------------

# set up data
base_folder = r'D:\Data\George\Projects\SpikeSorting\Neuroseeker\\' + \
                  r'Neuroseeker_2016_12_17_Anesthesia_Auditory_DoubleProbes\AngledProbe\KilosortResults'

indices_p = np.load(os.path.join(base_folder, r'indices_p.npy'))
values_p = np.load(os.path.join(base_folder, r'values_p.npy'))

n = indices_p.shape[0]
num_of_dims = 2
t_sne = np.random.random((n, num_of_dims))

delta = gpu.compute_gradient(t_sne, indices_p, values_p, verbose=True)







n = 1e6
spike_distances_on_probe = 8000 * np.random.random(n)
spike_indices_sorted_by_probe_distance = np.array([b[0] for b in sorted(enumerate(spike_distances_on_probe),
                                                                        key=lambda dist: dist[1])])
spike_distances_on_probe_sorted = np.array([b[1] for b in sorted(enumerate(spike_distances_on_probe),
                                                                 key=lambda dist: dist[1])])

t_sne_positions = np.random.random((n, 2)) * 0.0001

# cut the data into matrices to do distance calcs
distance_threshold = 100
max_elements_in_matrix = 2e9

indices_of_first_arrays, indices_of_second_arrays = \
    spikes.define_all_spike_spike_matrices_for_distance_calc(spike_distances_on_probe_sorted,
                                                             max_elements_in_matrix=max_elements_in_matrix,
                                                             probe_distance_threshold=distance_threshold)

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- TEST HIGH DIM SPACE DISTANCES CALCULATIONS -----------------------------------------------
# ----- generate some data ------
m = 100000
n = 20000
k = 500
per = 100
b_start_index = 0

print('Loading initial data onto GPU')
s = timer()
a = np.array(np.random.random((m, k)), dtype=np.float32)
b = np.array(np.random.random((n, k)), dtype=np.float32)
temp = np.array(np.zeros((m, n), dtype=np.float32))
distances_on_gpu = cuda.to_device(np.asfortranarray(temp))
e = timer()
data_time = e - s
print("LOADING DATA TIME:", "%.3f" % data_time, "s")


gpu._calculate_distances_on_gpu(a=a, b=b, distances_on_gpu=distances_on_gpu, verbose=True)

# -------- USING GEMM, GUVECTORIZE and d = sqrt(||a||^2 + ||b||^2 -2<a,b>) ----------
blas = cublas.Blas()


@guvectorize([(float32[:, :], float32[:])], '(m,k)->(m)', nopython=True)
def create_dot_product(a, dots_a):
    for i in np.arange(a.shape[0]):
        dots_a[i] = np.dot(a[i, :], a[i, :])


@guvectorize([(float32[:], float32[:], float32[:, :])], '(m),(n)->(m,n)', nopython=True)
def sums_of_dots(dots_a, dots_b, s_o_dots):
    for i in np.arange(dots_a.shape[0]):
        for j in np.arange(dots_b.shape[0]):
            s_o_dots[i, j] = dots_a[i] + dots_b[j]

@cuda.jit
def sums_of_dots_gpu(dots_a, dots_b, s_o_dots):
    a_index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    if a_index < dots_a.shape[0]:
        for b_index in range(dots_b.shape[0]):
            s_o_dots[a_index, b_index] = dots_a[a_index] + dots_b[b_index]


s = timer()

# calculate the inner product of each row
sd = timer()
dots_a = create_dot_product(a)
dots_b = create_dot_product(b)
ed = timer()
dot_time = ed - sd
print("MAKING THE DOT PRODUCTS TIME:", "%.3f" % dot_time, "s")

# calculate the ||a||^2 + ||b||^2 sum of dot products matrix (a.a + b.b) on the gpu and save on the gpu_temp matrix
sd = timer()
ddots_a = cuda.to_device(np.asfortranarray(dots_a))
ddots_b = cuda.to_device(np.asfortranarray(dots_b))

threadsperblock = 512
blockspergrid = math.ceil(ddots_a.shape[0] / threadsperblock)
sums_of_dots_gpu[blockspergrid, threadsperblock](ddots_a, ddots_b, distances_on_gpu)
numba.cuda.synchronize()
ed = timer()
s_o_dot_time = ed - sd
print("SUMMING THE DOT PRODUCTS ON THE GPU TIME:", "%.3f" % s_o_dot_time, "s")

# calculate the -2<a,b> cross dot products matrix and do the sum (||a||^2 + ||b||^2) -2<a,b>
sd = timer()
da = cuda.to_device(np.asfortranarray(a))
db = cuda.to_device(np.asfortranarray(b))
blas.gemm('N', 'T', m, n, k, -2.0, da, db, 1.0, distances_on_gpu)
numba.cuda.synchronize()
ed = timer()
time_gemm = ed - sd
print("CUDA BLAS GEMM TIME:", "%.3f" % time_gemm, "s")


# copy to host and square root to calculate the euclidean distances d = sqrt(||a||^2 + ||b||^2 -2<a,b>)
sd = timer()
temp = distances_on_gpu.copy_to_host()
dist_gemm = np.sqrt(temp)
ed = timer()
time_dist = ed - sd
print("LOAD ONTO HOST AND SQUARE ROOT TIME:", "%.3f" % time_dist, "s")

e = timer()
total_time = e - s
print("TOTAL ALL DISTANCES TIME:", "%.3f" % total_time, "s")


# sort and grab the smallest 300
s = timer()
radix_sort = sorting.RadixSort(maxcount=n, dtype=np.float32)
sorted_distances = np.empty((m, 3 * per))
sorted_spike_indices = np.empty((m, 3 * per))
for i in np.arange(m):
    keys = np.ascontiguousarray(dist_gemm[i])
    values = np.arange(keys.shape[0])
    radix_sort.sort(keys=keys, vals=values)
    sorted_distances[i, :] = keys[:3 * per]
    sorted_spike_indices[i, :] = values[:3 * per]
e = timer()
sort_time = e - s
print("SORTING TIME:", "%.3f" % sort_time, "s")



# sort and grab the smallest 300
s = timer()
radix_sort = sorting.RadixSort(maxcount=n, dtype=np.float32)
sorted_distances = np.empty((m, 3 * per))
sorted_spike_indices = np.empty((m, 3 * per))
for i in np.arange(10):
    #keys = np.ascontiguousarray(distances_on_gpu[i])
    keys = distances_on_gpu[i].copy_to_host()
    values = numba.types.uint32(np.array(np.arange(n) + 0))
    #values_on_gpu = cuda.to_device(values)
    radix_sort.sort(keys=keys, vals=values)
    sorted_distances[i, :] = keys[:3 * per]
    sorted_spike_indices[i, :] = values[:3 * per]
e = timer()
sort_time = e - s
print("SORTING TIME:", "%.3f" % sort_time, "s")


# ------Testing Low DImensional DIstances on GPU

m = 1000
n = 300  # per*3
dim = 3  # 2 or 3








# ============================  NOT USED !!!! ==========================================================================
# ======================================================================================================================

# ------- USING SCIPY's CDIST (CPU) -----------------------
s = timer()
dist_sp = spdist.cdist(a, b, 'euclidean')
e = timer()
time_sp = e - s
print("CDIST TIME:", "%.3f" % time_sp, "s")

print("Testing discrepancy between GEMM and CDIST %1.3e" % (np.sqrt(((dist_gemm - dist_sp) ** 2).sum())))


# ------- OLD STUFF FOR METHOD 1 ---------------------------
@jit(nopython=True)
def final_distances(cross_dot_products, s_o_dots):
    return (s_o_dots + cross_dot_products)

@guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(m, n),(m, n)->(m,n)', nopython=True)
def final_distances(cross_dot_products, s_o_dots, distances):
    for i in np.arange(cross_dot_products.shape[0]):
        for j in np.arange(cross_dot_products.shape[1]):
            distances[i, j] = (cross_dot_products[i, j] + s_o_dots[i, j])


# calculate the ||a||^2 + ||b||^2 sum of dot products matrix (a.a + b.b)
sd = timer()
s_o_dots = sums_of_dots(numba.types.float32(dots_a), numba.types.float32(dots_b))
ed = timer()
s_o_dot_time = ed - sd
print("SUMMING THE DOT PRODUCTS TIME:", "%.3f" % s_o_dot_time, "s")


# Method 2 (DOES NOT WORK)
# ------- USING cuda.jit TO CALCULATE DISTANCES ONE AT A TIME ON GPU ---------------------------
TPB = 16

@cuda.jit
def fast_eucledian_distance(da, db, dc):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= dc.shape[0] and y >= dc.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = da[x, ty + i * TPB]
        sB[ty, tx] = db[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial distance on shared memory
        for j in range(TPB):
            #tmp += sA[tx, j] * sB[j, ty] # THAT WORKS
            dif = sA[tx, j] - sB[j, ty] # THE EUCLIDEAN DISTANCE DOESN'T
            tmp += dif*dif
        # Wait until all threads finish computing
        cuda.syncthreads()

    dc[x, y] = tmp

s = timer()

da = cuda.to_device(np.asfortranarray(a))
db = cuda.to_device(np.asfortranarray(b))
dc = cuda.to_device(np.asfortranarray(temp))

threadsperblock = (16, 16)
blockspergrid_x = math.ceil(dc.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(dc.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
fast_eucledian_distance[blockspergrid, threadsperblock](da, db, dc)
numba.cuda.synchronize()
dist_cudajit = np.sqrt(dc.copy_to_host())

e = timer()
time_cudajit = e - s
print("CUDA.JIT TIME:", "%.3f" % time_cudajit, "s")
print("Testing discrepancy between CUDA.JIT and CDIST %1.3e" % (np.sqrt(((dist_cudajit - dist_sp) ** 2).sum())))



