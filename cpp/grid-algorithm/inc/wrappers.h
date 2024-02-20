#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "./cuda_functions.h"

#ifndef WRAPPERS_H
#define WRAPPERS_H

/********************
* Wrapper Functions *
********************/

void computeDensitiesWrapper(float *vecX, float *vecY, float *vecZ, float *output,
                         int startX, int endX, int startY, int endY,
                         int vecXYSize, int vecZSize)
{
  /*
  Wrap the operations needed to compute the densities of the likelihood function.
  */

  /* It seems that we are reinventing the wheel a bit as we could use the
  cuTENSOR library. This, however, has a steeeeep learning curve 😕
  */

  float *gridX, *gridY, *gridZ;
  int gridSize = vecXYSize * vecXYSize * vecZSize;
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&gridZ, gridSize * sizeof(float));

  linspaceCuda(vecX, vecXYSize, startX, endX);
  linspaceCuda(vecY, vecXYSize, startY, endY);

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
  */

  double **likesMatrix;
  int rows = likesSize;  // 101x101 rows
  int cols = densitiesSize / likesSize;  // of 50 elements each

  if (densitiesSize != rows * cols) {
    printf("ERROR: likesSize != rows * cols\n");
    return;
  }

  cudaMallocManaged(&likesMatrix, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    cudaMallocManaged(&likesMatrix[i], cols * sizeof(double));
  }

  reshapeArray<float, double>(densities, likesMatrix, cols, rows);
  computeLikesCuda(likes, likesMatrix, rows, cols);

  for (int i = 0; i < cols; i++) {
    cudaFree(likesMatrix[i]);
  }
  cudaFree(likesMatrix);
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
  */
  std::cout << "vector first element: " << posterior[0] << std::endl;

  // We start by creating the matrices for the reduction. We need to create two
  // because `marginalize` will modify them to perform the reduction making them
  // not reusable.

  int side = std::sqrt(posterior.size());

  double **posteriorMatrixMu = createMatrix(side, side);
  double **posteriorMatrixSigma = createMatrix(side, side);

  // Now we reshape the posterior vector. We make a host_vector which is more
  // flexible to work with
  thrust::host_vector<double> h_posterior = posterior;

  double *posteriorPtr = h_posterior.data();
  reshapeArray<double, double>(posteriorPtr, posteriorMatrixMu, side, side);
  reshapeArray<double, double>(posteriorPtr, posteriorMatrixSigma, side, side);

  std::cout << "Matrix first element: " << posteriorMatrixMu[0][0] << std::endl;

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