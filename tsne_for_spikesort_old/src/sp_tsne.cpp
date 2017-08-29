/*
*
* Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the Delft University of Technology.
* 4. Neither the name of the Delft University of Technology nor the names of
*    its contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
* EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
* IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
*
*/

#define CUB_STDERR

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdint.h>

#include "sptree.h"
#include "sp_tsne.h"

using namespace std;

// Perform t-SNE
void TSNE::run(double* Y, int N, int no_dims, unsigned int* col_P, double* val_P, int K,
	int perplexity, double theta, double eta, int iterations, int verbose)
{
	//setbuf(stdout, NULL);
	setvbuf(stdout, NULL, _IONBF, 1024);

    if (verbose > 0) printf("Read the %i x %i data matrices successfully!\n", N, K);
	// Determine whether we are using an exact algorithm
	if (N - 1 < 3 * perplexity) {
		printf("Perplexity ( = %i) too large for the number of data points (%i)!\n", perplexity, N);
		exit(1);
	}
	if (verbose > 0) printf("Using no_dims = %d, perplexity = %d, learning rate = %f, and theta = %f\n", no_dims, perplexity, eta, theta);



	// Set learning parameters
	float total_time = .0;
	clock_t start, end;
	int max_iter = iterations, stop_lying_iter = 250, mom_switch_iter = 250;
	double momentum = .5;
	double final_momentum = .8;
	float exageration = 12.0;
	unsigned int* row_P = NULL;

    int* temp_P = (int*)malloc((N + 1) * sizeof(int));
	if (*temp_P == NULL) { printf("Memory allocation failed 0!\n"); exit(1); }

	row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	if (*row_P == NULL) { printf("Memory allocation failed 2!\n"); exit(1); }
	row_P[0] = 0;
	for (int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int)K;

	// Allocate some memory
	double* dY = (double*)malloc(N * no_dims * sizeof(double));
	double* uY = (double*)malloc(N * no_dims * sizeof(double));
	double* gains = (double*)malloc(N * no_dims * sizeof(double));
	if (dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed 1!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++)    uY[i] = .0;
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;


	double* cur_P = (double*)malloc((N - 1) * sizeof(double));
	if (cur_P == NULL) { printf("Memory allocation failed 3!\n"); exit(1); }



	// Symmetrize input similarities
	symmetrizeMatrix(&row_P, &col_P, &val_P, N);
	double sum_P = .0;
	for (unsigned int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
	for (unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;


	// Lie about the P-values
	for (unsigned int i = 0; i < row_P[N]; i++)		val_P[i] *= exageration;

	// Initialize solution (randomly)
	for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;


	// Perform main training loop
	if (verbose > 0) printf("\nLearning embedding...\n");
	start = clock();
	for (int iter = 0; iter < max_iter; iter++) {

		// Compute (approximate) gradient
		computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta);

		// Update gains
		//for(int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
		for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .05) : (gains[i] * .95);
		for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

		// Perform gradient update (with momentum and gains)
		for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		for (int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

		// Make solution zero-mean
		zeroMean(Y, N, no_dims);

		// Stop lying about the P-values after a while, and switch momentum
		if (iter == stop_lying_iter) {
			for (unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= exageration;
		}
		if (iter == mom_switch_iter) momentum = final_momentum;

		// Save tSNE progress after each iteration
		if (verbose > 2)
		{
			// Open file, write first 2 integers and then the data
			FILE *h;
			char interim_filename[_MAX_PATH];
			sprintf_s(interim_filename, "interim_%06i.dat", iter);
			fopen_s(&h, interim_filename, "w + b");
			if (h == NULL)
			{
				printf("Error: could not open data file.\n");
				return;
			}
			fwrite(&N, sizeof(int), 1, h);
			fwrite(&no_dims, sizeof(int), 1, h);
			fwrite(Y, sizeof(double), N * no_dims, h);
			fclose(h);
		}

		// Print out progress
		if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
			end = clock();
			double C = .0;
			C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
			if (iter == 0) {
				if (verbose > 1) printf("Iteration %d: error is %f\n", iter + 1, C);
			}
			else {
				total_time += (float)(end - start) / CLOCKS_PER_SEC;
				if (verbose > 1) printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
			}
			start = clock();
		}
	}
	end = clock(); total_time += (float)(end - start) / CLOCKS_PER_SEC;

	// Clean up memory
	free(dY);
	free(uY);
	free(gains);
	//free(row_P); row_P = NULL;
	//free(col_P); col_P = NULL;
	//free(val_P); val_P = NULL;

	if (verbose > 0) printf("Fitting performed in %4.2f seconds.\n", total_time);
}



// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
	// Construct space-partitioning tree on current map
	SPTree* tree = new SPTree(D, Y, N);

	// Compute all terms required for t-SNE gradient
	double sum_Q = .0;
	double* pos_f = (double*)calloc(N * D, sizeof(double));
	double* neg_f = (double*)calloc(N * D, sizeof(double));
	if (pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed 4!\n"); exit(1); }


	tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for (int i = 0; i < N * D; i++) {
		dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

	// Get estimate of normalization term
	SPTree* tree = new SPTree(D, Y, N);
	double* buff = (double*)calloc(D, sizeof(double));
	double sum_Q = .0;
	for (int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

	// Loop over all edges to compute t-SNE error
	int ind1, ind2;
	double C = .0, Q;
	for (int n = 0; n < N; n++) {
		ind1 = n * D;
		for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
			Q = .0;
			ind2 = col_P[i] * D;
			for (int d = 0; d < D; d++) buff[d] = Y[ind1 + d];
			for (int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
			for (int d = 0; d < D; d++) Q += buff[d] * buff[d];
			Q = (1.0 / (1.0 + Q)) / sum_Q;
			C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
		}
	}

	// Clean up memory
	free(buff);
	delete tree;
	return C;
}

// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

	// Get sparse matrix
	unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
	double* val_P = *_val_P;

	// Count number of elements and row counts of symmetric matrix
	int* row_counts = (int*)calloc(N, sizeof(int));
	if (row_counts == NULL) { printf("Memory allocation failed 5!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) present = true;
			}
			if (present) row_counts[n]++;
			else {
				row_counts[n]++;
				row_counts[col_P[i]]++;
			}
		}
	}
	int no_elem = 0;
	for (int n = 0; n < N; n++) no_elem += row_counts[n];

	// Allocate memory for symmetrized matrix
	unsigned int* sym_row_P = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
	unsigned int* sym_col_P = (unsigned int*)malloc(no_elem * sizeof(unsigned int));
	double* sym_val_P = (double*)malloc(no_elem * sizeof(double));
	if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed 6!\n"); exit(1); }

	// Construct new row indices for symmetric matrix
	sym_row_P[0] = 0;
	for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int)row_counts[n];

	// Fill the result matrix
	int* offset = (int*)calloc(N, sizeof(int));
	if (offset == NULL) { printf("Memory allocation failed 7!\n"); exit(1); }
	for (int n = 0; n < N; n++) {
		for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

			// Check whether element (col_P[i], n) is present
			bool present = false;
			for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
				if (col_P[m] == n) {
					present = true;
					if (n <= col_P[i]) {                                                 // make sure we do not add elements twice
						sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
						sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
						sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
						sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
					}
				}
			}


			// If (col_P[i], n) is not present, there is no addition involved
			if (!present) {
				sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
				sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
				sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
				sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
			}


			// Update offsets
			if (!present || (present && n <= col_P[i])) {
				offset[n]++;
				if (col_P[i] != n) offset[col_P[i]]++;
			}
		}
	}

	// Divide the result by two
	for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

	// Return symmetrized matrices
	free(*_row_P); *_row_P = sym_row_P;
	free(*_col_P); *_col_P = sym_col_P;
	free(*_val_P); *_val_P = sym_val_P;

	// Free up some memery
	free(offset); offset = NULL;
	free(row_counts); row_counts = NULL;
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*)calloc(D, sizeof(double));
	if (mean == NULL) { printf("Memory allocation failed 8!\n"); exit(1); }
	int nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
		nD += D;
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double)N;
	}

	// Subtract data mean
	nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
		nD += D;
	}
	free(mean); mean = NULL;
}

void TSNE::normalize(double* X, int N, int D){
	double max_X = .0;
	for (int i = 0; i < N * D; i++) {
		if (X[i] > max_X) max_X = X[i];
	}
	for (int i = 0; i < N * D; i++) X[i] /= max_X;
}


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity,
	double* eta, int* iterations, int* seed, float* gpu_mem, int* verbose, int* rand_seed) {

	// Open file, read first 2 integers, allocate memory, and read the data
	FILE *h;
	if ((h = fopen("data.dat", "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
	fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
	fread(eta, sizeof(double), 1, h);										// eta (learning rate)
	fread(no_dims, sizeof(int), 1, h);										// output dimensionality
	fread(iterations, sizeof(int), 1, h);									// number of iterations
	fread(seed, sizeof(int), 1, h);											// number of samples to t-sne
	fread(gpu_mem, sizeof(float), 1, h);									// percentage of gpu memory to use (if 0 no gpu is used)
	fread(verbose, sizeof(int), 1, h);										// verbosity (between 0 and 2)
	*data = (double*)malloc(*d * *n * sizeof(double));
	if (*data == NULL) { printf("Memory allocation failed 9!\n"); exit(1); }
	fread(*data, sizeof(double), *n * *d, h);                               // the data
	if (!feof(h)) fread(rand_seed, sizeof(int), 1, h);                      // random seed
	fclose(h);
	if (*verbose > 0) printf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, double* costs, int n, int d, int verbose) {

	// Open file, write first 2 integers and then the data
	FILE *h;
	if ((h = fopen("result.dat", "w+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
	fwrite(data, sizeof(double), n * d, h);
	fwrite(costs, sizeof(double), n, h);
	fclose(h);
	if (verbose > 0) printf("Wrote the %i x %i data matrix successfully!\n\n", n, d);
}

