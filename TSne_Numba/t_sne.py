import numpy as np
import sys


def run():
    # find distances in hd space and sort

    # compute_gaussian_perplexity

    # symmetrize_matrix and renormalize

    # lie about p-values and initialize solution

    # run loop
        # compute_gradient
        # update gains and update gradient
        # zero mean solution
        # evaluate error and print progress
    pass


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