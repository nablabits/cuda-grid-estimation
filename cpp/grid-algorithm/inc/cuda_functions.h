/* Handy function to create arrays */
#include <cuda_runtime.h>

#include "../inc/kernels.h"
#include "../inc/utils.h"

#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H


void generateNormalCuda(unsigned int n, float mu, float sigma,
                        float *observations) {
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


void linspaceCuda(float* array, int size, float start, float end) {
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
  /* Define the configuration of the create3dGrid kernel. */
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
  /* Define the configuration of the computeDensitiesKernel. */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeDensitiesKernel<<<numBlocks, threadsPerBlock>>>(densities, gridX,
                                                         gridY, gridZ, gridSize);

  cudaDeviceSynchronize();

  // TODO: remove these guys once we are done with the full algorithm
  printArray(gridX, 10);
  printArray(gridY, 10);
  printArray(gridZ, 10);
  printArray(densities, 10);
}


void computeLikesCuda(double *likes, double **likesMatrix, int rows, int cols)
{
  /* Define the configuration of the computeLikesKernel. */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((rows * cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeLikesKernel<<<numBlocks, threadsPerBlock>>>(likes, likesMatrix, rows, cols);

  // Always synchronize before printing data.
  cudaDeviceSynchronize();

  // TODO: remove these guys once we are done with the full algorithm
  printf("------->\n");
  printArrayd(likes, 10);
}

void computePosteriorCuda(double *likes, double *posterior, int likesSize)
{

  double *sum;
  cudaMallocManaged(&sum, sizeof(double));

  double *likesCopy = new double;
  for (int i = 0; i < likesSize; i++) {
    *likesCopy += likes[i];
  }

  printf("this is likes copy: %e\n", *likesCopy);

  int threadsPerBlock = 256;  // the number of threads per block
  int numBlocks = (likesSize + threadsPerBlock - 1) / threadsPerBlock;  // effectively, 4096 blocks
  printf("kernel size: %dx%d\n", numBlocks, threadsPerBlock);

  computePosteriorKernel<<<numBlocks, threadsPerBlock>>>(likes, sum, likesSize);

  cudaDeviceSynchronize();

  printf("this is the sum: %e\n", *sum);
  printf("this is likes[0]: %e\n", likes[0]);

  // we can also try the thrust library

  delete [] likesCopy;
  cudaFree(sum);
}
#endif