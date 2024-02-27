#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "./cuda_functions.h"
#include "./utils.h"

#ifndef WRAPPERS_H
#define WRAPPERS_H

/********************
* Wrapper Functions *
********************/

void computeDensitiesWrapper(float *vecX, float *vecY, float *vecZ, 
                             float *output, int startX, int endX, int startY, 
                             int endY, int vecXYSize, int vecZSize)
{
  /*
  Wrap the operations to compute the densities of the likelihood function.

  It seems that we are reinventing the wheel a bit as we could use the
  cuTENSOR library. This, however, has a steeeeep learning curve 😕

  Arguments:
    vecX: the array that will contain the mus (101).
    vecY: the array that will contain the sigmas (101).
    vecZ: the array containing the observations (50).
    output: the array where the densities will be stored (101x101x50)
    startX: the start value for the mu range in vecX
    endX: the end value for the mu range in vecX
    startY: the start value for the sigma range in vecY
    endY: the end value for the sigma range in vecY
    vecXYSize: the size of either vecX or vecY, 101
    vecZSize: the size of the array of observations (50)
  */

  /***************************************
  * Create the Ranges for the Parameters *
  ***************************************/

  /*
  We start by creating the ranges for the parameters using the equivalent of
  np.linspace. This range will allow us to create the outer product in the grid
  step.
  */


  float *gridX, *gridY, *gridZ;
  int gridSize = vecXYSize * vecXYSize * vecZSize;
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&gridZ, gridSize * sizeof(float));

  linspaceCuda(vecX, vecXYSize, startX, endX);
  linspaceCuda(vecY, vecXYSize, startY, endY);

  /*******************
  * Create the Grids *
  *******************/

  /*
  We need a grid with the outer product of three vectors that will represent the
  combinations of the parameters we want to estimate plus the observations. This
  is, if our vectors are [1, 2, 3] & [4, 5 ,6] & [1, 2] then our outer product
  will be:

  [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
  [4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]
  [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

  We use extensively the word `grid`, but it won't be a proper grid as the
  output will be a vector. Granted, a vector with the same elements as the grid,
  v1 x v2 x v3, but a vector after all.
  */

  create3dGridCuda(
    vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize
  );

  checkArrays(gridX, gridY, gridZ);

  computeDensitiesCuda(output, gridX, gridY, gridZ, gridSize);

  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}


void computeLikesWrapper(float *densities, double *likes, int densitiesSize, int likesSize)
{
  /*
  Wrap the operations needed to compute the likelihood function.

  Arguments:
    densities: the array of densities for each mu, sigma and obs. (101x101x50)
    likes: the array where we will store the likelihoods.
    densitiesSize: the size of the densities array (101x101x50)
    likesSize: the size of the likelihood array (101x101)
  */

  int rows = likesSize;  // 101x101 rows
  int cols = densitiesSize / likesSize;  // of 50 elements each

  if (densitiesSize != rows * cols) {
    printf("ERROR: likesSize != rows * cols\n");
    return;
  }

  double **likesMatrix = allocateMatrix(rows, cols);

  reshapeArray<float, double>(densities, likesMatrix, cols, rows);
  computeLikesCuda(likes, likesMatrix, rows, cols);

  freeMatrix(likesMatrix, rows);
}


double* computeExpectationsWrapper(thrust::device_vector<double> &posterior,
                                float *vectorMu, float *vectorSigma)
{
  /*
  Wrap the operations needed to extract the marginals from the grid and compute
  their expectations

  We have a thrust vector of 10200 elements and we need to convert it to a
  matrix of 101x101 so we can compute the sum over rows (marginal of mu) and the
  sum over columns (marginal of sigma).

  Once we have the marginals we can compute the expected value for each array

  Arguments:
    posterior: the thrust vector containing the posteriors (10201)
    vectorMu: the array where we will store the marginals of mu (101)
    vectorSigma: the array where we will store the marginals of sigma (101)

  Returns:
    An array of two elements containing the expectation of mu and sigma.
  */

  // We start by creating the matrices for the reduction. We need to create two
  // because `marginalize` will modify them to perform the reduction making them
  // not reusable.

  int side = std::sqrt(posterior.size());

  double **posteriorMatrixMu = allocateMatrix(side, side);
  double **posteriorMatrixSigma = allocateMatrix(side, side);

  // Now we reshape the posterior vector. We make a host_vector which is more
  // flexible to work with
  thrust::host_vector<double> h_posterior = posterior;

  double *posteriorPtr = h_posterior.data();
  reshapeArray<double, double>(posteriorPtr, posteriorMatrixMu, side, side);
  reshapeArray<double, double>(posteriorPtr, posteriorMatrixSigma, side, side);

  // Define the arrays we will use to store the marginals.
  double *marginalMu, *marginalSigma;
  cudaMallocManaged(&marginalMu, side * sizeof(double));
  cudaMallocManaged(&marginalSigma, side * sizeof(double));

  // Compute the marginals
  marginalizeCuda(marginalMu, posteriorMatrixMu, side, side, 1);
  marginalizeCuda(marginalSigma, posteriorMatrixSigma, side, side, 0);

  // Finally, compute the expectations
  static double expectations[2] = {
    computeExpectationsCuda(marginalMu, vectorMu, side),
    computeExpectationsCuda(marginalSigma, vectorSigma, side)
  };

  // Free up the memory
  freeMatrix(posteriorMatrixMu, side);
  freeMatrix(posteriorMatrixSigma, side);
  cudaFree(marginalMu);
  cudaFree(marginalSigma);

  return expectations;
}

#endif