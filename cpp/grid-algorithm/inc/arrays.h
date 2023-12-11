/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H


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
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    linspaceKernel<<<gridSize, blockSize>>>(array, size, start, end);
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
  dim3 numBlocks_((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  createGridKernel<<<numBlocks_, threadsPerBlock>>>(
    vectorX, vectorY, gridX, gridY, size
  );

  cudaDeviceSynchronize();
}

#endif