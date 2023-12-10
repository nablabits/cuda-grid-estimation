/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H


__global__ void linspaceKernel(float *array, int size, float start, float end) {
  /*
  Create a linearly spaced array on the device.

  array: output array
  n: length of array
  start: start of range
  end: end of range
  */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = start + idx * (end - start) / (size - 1);
    }
}

void linspaceCuda(float* array, int size, float start, float end) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    linspaceKernel<<<gridSize, blockSize>>>(array, size, start, end);
    cudaDeviceSynchronize();
}


__global__ void createGridKernel(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size
  )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;

  int pos = i * size + ii;
  if (i < size && ii < size) {
    gridX[pos] = vectorX[i];
    gridY[pos] = vectorY[ii];
  }
}

void createGrid(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size,
  float startMu, float endMu, float startSigma, float endSigma
  )
{
  linspaceCuda(vectorX, size, startMu, endMu);
  linspaceCuda(vectorY, size, startSigma, endSigma);

  dim3 threadsPerBlock(16, 16);  // You can adjust the block size as needed
  dim3 numBlocks_((101 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (101 + threadsPerBlock.y - 1) / threadsPerBlock.y);

  createGridKernel<<<numBlocks_, threadsPerBlock>>>(vectorX, vectorY, gridX, gridY, 101);

  cudaDeviceSynchronize();

}

#endif