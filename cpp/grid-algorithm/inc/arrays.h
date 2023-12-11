/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H

void printArray(float *arr, int n=3) {
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

__global__ void linspaceKernel(float *array, int size, float start, float end) {
  /*
  Create a linearly spaced array on the device.

  array: output array
  n: length of array
  start: start of range (included)
  end: end of range (included)
  */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = start + idx * (end - start) / (size - 1);
    }
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


__global__ void createGridKernel(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size
  )
{
  /*
  Create the outer product of two verctors on the device.

  We need a grid with the outer product of two vectors that will represent the
  combinations of the parameters we want to estimate. This is, if our vectors
  are [1, 2, 3] & [4, 5 ,6], then our outer product will be:

  [1, 2, 3, 1, 2, 3, 1, 2, 3]
  [4, 4, 4, 5, 5, 5, 6, 6, 6]

  TODO: we treat vectors and grids as separate objects. A possible improvement
  here could be to treat them as a single multidimensional array.

  Arguments:
    vectorX: the first vector
    vectorY: the second vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    size: the size of the vectors
  */

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;

  int pos = i * size + ii;
  if (i < size && ii < size) {
    gridX[pos] = vectorX[i];
    gridY[pos] = vectorY[ii];
  }
}

void createGridCuda(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size
  )
{
  /*
  Define the configuration of createGrid kernel and call it.

  Arguments:
    vectorX: the first vector
    vectorY: the second vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    size: the size of the vectors
  */

  dim3 threadsPerBlock(16, 16);  // You can adjust the block size as needed
  dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  createGridKernel<<<numBlocks, threadsPerBlock>>>(
    vectorX, vectorY, gridX, gridY, size
  );

  cudaDeviceSynchronize();
}

float normalPdf(float x, float mu, float sigma) {
  float result = 1.0f / (sigma * sqrt(2.0f * M_PI)) * exp(-0.5f * pow((x - mu) / sigma, 2.0f));
  return result;
}

void simpleLikelihood() {
  float observations = 20.5f;
  int vecSize = 3;
  int gridSize = vecSize * vecSize;

  float *vecX, *vecY;
  float *gridX, *gridY, *likes;
  cudaMallocManaged(&vecX, vecSize * sizeof(float));
  cudaMallocManaged(&vecY, vecSize * sizeof(float));
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&likes, gridSize * sizeof(float));

  linspaceCuda(vecX, 3, 19.0f, 21.0f);
  linspaceCuda(vecY, 3, 1.0f, 3.0f);
  createGridCuda(vecX, vecY, gridX, gridY, vecSize);

  for (int i = 0; i < gridSize; i++) {
    likes[i] = normalPdf(observations, gridX[i], gridY[i]);
  }

  printArray(gridX, gridSize);
  printArray(gridY, gridSize);
  printArray(likes, gridSize);

  // Clean up
  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(likes);
  cudaFree(vecX);
  cudaFree(vecY);
}

#endif