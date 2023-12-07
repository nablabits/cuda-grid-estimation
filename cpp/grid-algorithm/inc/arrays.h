/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H

// TODO: this should be a device function
void createGrid(float *vectorX, float *vectorY, float *gridX, float *gridY, int size) {
  int pos = 0;
  for (int i = 0; i < size; i++) {
    for (int ii = 0; ii < size; ii++) {
        gridX[pos] = vectorX[ii];
        gridY[pos] = vectorY[i];
        pos++;
    }
  }
}

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

#endif