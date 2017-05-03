import numpy as np
import sys
from TSne_Numba import gpu
import matplotlib.pylab as pylab
import time
from TSne_Numba import sptree_jit as sptree


def run(data, indices_of_first_and_second_matrices, iters, perplexity, eta=200, num_dims=2, theta=0.2, verbose=True):

    # zero mean input data
    data = pylab.demean(data, axis=0)

    # normalize input data
    data /= data.max()

    momentum = 0.5
    final_momentum = 0.8
    exaggeration = 12.0
    stop_lying_iter = 250
    mom_switch_iter = 250

    num_samples = data.shape[0]

    # find distances in hd space and sort
    s1 = time.time()
    closest_indices_in_hd, closest_distances_in_hd = \
        gpu.calculate_knn_distances_close_on_probe(template_features_sorted=data,
                                                   indices_of_first_and_second_matrices=
                                                   indices_of_first_and_second_matrices,
                                                   perplexity=perplexity,
                                                   verbose=verbose)
    e1 = time.time()
    if verbose > 1:
        print('Time for Knn distance calculation: ' + str(e1 - s1))

    # compute_gaussian_perplexity
    indices_p, values_p = _compute_gaussian_perplexity(closest_indices_in_hd, closest_distances_in_hd,
                                                       perplexity=perplexity)

    # symmetrize_matrix

    # renormalize
    sum_p = np.sum(values_p)
    values_p /= sum_p

    # lie about p-values
    values_p *= exaggeration

    # initialize solution
    y = np.random.random((num_samples, num_dims)) * 0.0001
    uy = np.zeros((num_samples, num_dims))
    gains = np.ones((num_samples, num_dims))

    # run loop
    verbose_gradient = False
    if verbose > 2:
        verbose_gradient = True
    s2 = time.time()
    s3 = time.time()
    for it in np.arange(iters):
        # compute_gradient
        dy = gpu.compute_gradient_on_gpu(y, indices_p=indices_p, values_p=values_p, verbose=verbose_gradient)
        # dy = _compute_gradient_on_cpu_with_sptree(y, indices_p=indices_p, values_p=values_p, theta=theta)

        # update gains
        gains[np.argwhere(np.sign(dy) != np.sign(uy))] += 0.05
        gains[np.argwhere(np.sign(dy) == np.sign(uy))] *= 0.95
        gains[np.argwhere(gains < 0.01)] = 0.01

        # update gradient
        uy = momentum * uy - eta * gains * dy
        y += uy

        # zero mean solution
        y = pylab.demean(y, axis=0)

        if it == stop_lying_iter:
            values_p /= exaggeration
        if it == mom_switch_iter:
            momentum = final_momentum

        # evaluate error and print progress
        if it % 5 == 0 and verbose:
            e3 = time.time()
            print('Time for iteration ' + str(it) + ' = ' + str(e3 - s3))
            s3 = time.time()

    e2 = time.time()
    if verbose > 1:
        print('Time for calculating the t-sne data: ' + str(e2 - s2))
    if verbose > 1:
        print('Time for total calculation: ' + str(e2 - s1))

    return y


def _compute_gaussian_perplexity(selected_sorted_indices, selected_sorted_distances,
                                 perplexity=100):
    k = selected_sorted_indices.shape[1]
    n = selected_sorted_indices.shape[0]
    dbl_min = sys.float_info[3]
    dbl_max = sys.float_info[0]

    ind_p = selected_sorted_indices.astype(np.int)
    val_p = np.empty((n, k))
    for spike in np.arange(n):
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

    return ind_p, val_p


def _compute_gradient_on_cpu_with_sptree(t_sne, indices_p, values_p, theta):

    dimension = t_sne.shape[1]
    num_of_points = t_sne.shape[0]
    neg_forces = np.zeros((num_of_points, dimension))
    sum_q = [0.0]

    tree = sptree.SPTree(inp_dimension=dimension, inp_data=t_sne, inp_num_of_points=num_of_points)

    pos_forces = tree.compute_edge_forces(indices_p=indices_p, values_p=values_p, N=num_of_points)
    for n in np.arange(num_of_points):
        tree.compute_non_edge_forces(point_index=n, theta=theta, neg_force=neg_forces, sum_q=sum_q)

    dy = pos_forces - (neg_forces / sum_q[0])

    return dy