# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "sp_tsne.h":
    cdef cppclass TSNE:
        TSNE()
        void run(double* Y, int N, int no_dims, unsigned int* col_P, double* val_P, int K,
		         int perplexity, double theta, double eta, int iterations, int verbose)


cdef class SP_TSNE:
    cdef TSNE* thisptr # hold a C++ instance

    def __cinit__(self):
        self.thisptr = new TSNE()

    def __dealloc__(self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, Y, N, no_dims, col_P, val_P, K, perplexity, theta, eta, iterations, verbose):
        #cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y)
        cdef np.ndarray[np.npy_uint, ndim=2, mode='c'] _col_P = np.ascontiguousarray(col_P)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _val_P = np.ascontiguousarray(val_P)
        self.thisptr.run(&_Y[0,0], N, no_dims, &_col_P[0,0], &_val_P[0,0], K,
                         perplexity, theta, eta, iterations, verbose)
        return _Y
