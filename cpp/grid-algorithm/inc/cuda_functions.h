/* Handy function to create arrays */
#include <cuda_runtime.h>

#include "../inc/kernels.h"
#include "../inc/utils.h"

#include <thrust/device_vector.h>

#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H


void generateNormalCuda(unsigned int n, float mu, float sigma,
                        float *observations)
{
  /*
  Define the configuration of the kernels to generate random variates.

  Arguments:
    n: the number of variates to generate.
    mu: the mean of the normal distribution.
    sigma: the standard deviation of the normal distribution.
    observations: the array where the random variates will be stored.
  */
  const unsigned int threadsPerBlock = 64;
  const unsigned int numBlocks = (n + threadsPerBlock  -1) / threadsPerBlock;

  curandState *devStates;

  // We need to assign states to each thread.
  cudaMallocManaged(&devStates,
                    threadsPerBlock * numBlocks * sizeof(curandState));

  setup_kernel<<<numBlocks, threadsPerBlock>>>(devStates);

  generate_normal_kernel<<<numBlocks, threadsPerBlock>>>(
    devStates, n, mu, sigma, observations
  );

  cudaDeviceSynchronize();

  // Clean up
  cudaFree(devStates);
}


void linspaceCuda(float* array, int size, float start, float end)
{
    /*
    Define the configuration of the linspace kernel.

    array: output array
    n: length of array
    start: start of range (included)
    end: end of range (included)
    */
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    linspaceKernel<<<numBlocks, threadsPerBlock>>>(array, size, start, end);
    cudaDeviceSynchronize();
}


void create3dGridCuda(float *vecX, float *vecY, float *vecZ,
                  float *gridX, float *gridY, float *gridZ,
                  int vecXYSize, int vecZSize)
{
  /*
  Define the configuration of the create3dGrid kernel.

  Arguments:
    vecX: the first vector
    vecY: the second vector
    vecZ: the third vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    gridZ: the grid of the third vector to be filled
    vecXYSize: the size of the vectors vecX and vecY
    vecZSize: the size of the vector vecZ
  */
  dim3 threadsPerBlock(4, 4, 4);
  dim3 numBlocks((vecXYSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (vecXYSize + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (vecZSize + threadsPerBlock.z - 1) / threadsPerBlock.z);

  create3dGridKernel<<<numBlocks, threadsPerBlock>>>(
    vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize
  );

  cudaDeviceSynchronize();
}


void computeDensitiesCuda(float *densities, float *gridX, float *gridY,
                          float *gridZ, int gridSize)
{
  /*
  Define the configuration of the computeDensitiesKernel.

  Arguments:
    densities: the array where we will store the computed densities.
    gridX: the array of values for mu as an outer product `mu x sigma x obs`
    gridY: the array of values for sigma as an outer product `mu x sigma x obs`
    gridZ: the array of values for the observations as an outer product
    gridSize: the size of all above arrays (101x101x50)
  */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeDensitiesKernel<<<numBlocks, threadsPerBlock>>>(densities, gridX,
                                                         gridY, gridZ, gridSize);

  cudaDeviceSynchronize();
}


void computeLikesCuda(double *likes, double **likesMatrix, int rows, int cols)
{
  /*
  Define the configuration of the computeLikesKernel.

  Arguments:
    likes: the array where we will store the likelihoods (101x101)
    likesMatrix: matrix of the outer product mu x sigma and the obs.
    rows: the number of rows of the matrix (10201)
    cols: the number of columns of the matrix (50)
  */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((rows * cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeLikesKernel<<<numBlocks, threadsPerBlock>>>(likes, likesMatrix, rows, cols);

  // Always synchronize before printing data.
  cudaDeviceSynchronize();
}


void computePosteriorCuda(thrust::device_vector<double>& likesV,
                          thrust::device_vector<double>& posteriorV)
{
  /*
  Compute the posterior using thrust library

  In the posterior we just need to normalize the likes vector as we are assuming
  a flat prior. Had we chosen some custom prior we would have needed to first
  compute joint prob between the prior and the likes before normalization as
  usual with Bayes theorem.

  Arguments:
    likesV: a device vector containing the likelihoods
    posteriorV: a device vector that will store the posteriors
  */

  double sum = thrust::reduce(likesV.begin(), likesV.end());

  // Now that we have the sum we can saxpy_fast posteriorV.
  saxpy_fast(1 / sum, likesV, posteriorV);
}

void marginalizeCuda(
  double* marginal, double**posteriorMatrix, int rows, int cols, int axis)
{
  /*
  Define the configuration of the marginalize Kernel.

  Arguments:
    marginal: the array where the marginal will be stored. (101)
    posteriorMatrix: the matrix that contains the posteriors
    rows: the number of rows in the posteriorMatrix (101)
    columns: the number of columns in the posteriorMatrix (101)
    axis: the axis over which the marginalization will be performed.
  */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((rows * cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

  marginalizeKernel<<<numBlocks, threadsPerBlock>>>(
    marginal, posteriorMatrix, rows, cols, axis
  );

  cudaDeviceSynchronize();
}


double computeExpectationsCuda(double *marginal, float *vector, int size)
{
  /*
  Compute the weighted average between an array of probs and an array of values.

  Arguments:
    marginal: the array that contains the marginals either mu or sigma. (101)
    vector: the linspace of ranges of mu or sigma. (101)
    size: the size of either each array. (101)

  Returns:
    The expected value.
  */

  // Thrust vectors will make the whole operation a piece of cake, so let's
  // create them.
  thrust::device_vector<double> marginalDvc(
    thrust::device_pointer_cast(marginal),
    thrust::device_pointer_cast(marginal + size)
  );

  thrust::device_vector<double> vectorDvc(
    thrust::device_pointer_cast(vector),
    thrust::device_pointer_cast(vector + size)
  );

  // take the product between the marginal and the values and store back in the
  // marginal.
  thrust::multiplies<double> binary_op;
  thrust::transform(marginalDvc.begin(), marginalDvc.end(),
                    vectorDvc.begin(), marginalDvc.begin(), binary_op);

  // Now reduce it.
  double expectation = thrust::reduce(marginalDvc.begin(), marginalDvc.end());

  return expectation;
}
#endif