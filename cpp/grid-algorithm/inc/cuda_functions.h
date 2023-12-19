/* Handy function to create arrays */
#include <cuda_runtime.h>

#include "../inc/kernels.h"
#include "../inc/utils.h"

#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H


void generateNormalCuda(unsigned int n, float mu, float sigma, float *observations) {
  const unsigned int threadsPerBlock = 64;
  const unsigned int numBlocks = (n + threadsPerBlock  -1) / threadsPerBlock;

  curandState *devStates;

  // We need to assign states to each thread.
  cudaMallocManaged(&devStates, threadsPerBlock * numBlocks * sizeof(curandState));

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

    TODO: `end` should not be included as to replicate the original
    implementation np.linspace.

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


void computeLikesCuda(float *likes, float *gridX, float *gridY, float *gridZ,
                      int gridSize)
{
  /* Define the configuration of the computeLikesKernel kernel. */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeLikesKernel<<<numBlocks, threadsPerBlock>>>(likes, gridX, gridY, gridZ,
                                                     gridSize);

  cudaDeviceSynchronize();

  // TODO: remove these guys once we are done with the full algorithm
  printArray(gridX, 10);
  printArray(gridY, 10);
  printArray(gridZ, 10);
  printArray(likes, 10);
}


void computeLikesWrapper(float *vecX, float *vecY, float *vecZ, float *output,
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

  // TODO: this is basically compute densities, as the likes is the prod over axis 2
  computeLikesCuda(output, gridX, gridY, gridZ, gridSize);

  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}

void computePosteriorWrapper(float *likes, double *posterior, int likesSize, int posteriorSize)
{
  /*
  Wrap the operations needed to compute the posterior function.
  */

  double **likesMatrix;
  int rows = posteriorSize;  // 101x101 rows
  int cols = likesSize / posteriorSize;  // of 50 elements each

  if (likesSize != rows * cols) {
    printf("ERROR: likesSize != rows * cols\n");
    return;
  }

  cudaMallocManaged(&likesMatrix, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    cudaMallocManaged(&likesMatrix[i], cols * sizeof(double));
  }

  reshapeArray(likes, likesMatrix, cols, rows);

  dim3 threadsPerBlock(256);
  dim3 numBlocks((likesSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computePosteriorKernel<<<numBlocks, threadsPerBlock>>>(posterior, likesMatrix, rows, cols);

  // Always syncronize before printing data.
  cudaDeviceSynchronize();

  printf("------->\n");
  printArrayd(posterior, 10);

  for (int i = 0; i < cols; i++) {
    cudaFree(likesMatrix[i]);
  }
  cudaFree(likesMatrix);
}

#endif