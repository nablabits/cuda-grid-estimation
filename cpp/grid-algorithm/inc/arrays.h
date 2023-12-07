/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H

// TODO: this should be a device function
void createGrid(int *vectorX, int *vectorY, int *gridX, int *gridY, int size) {
  int pos = 0;
  for (int i = 0; i < size; i++) {
    for (int ii = 0; ii < size; ii++) {
        gridX[pos] = vectorX[ii];
        gridY[pos] = vectorY[i];
        pos++;
    }
  }
}

__global__ void linspaceKernel(float *array, int n, float start, float end) {
  /*
  Create a linearly spaced array on the device.

  array: output array
  n: length of array
  start: start of range
  end: end of range
  */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = start + idx * (end - start) / (n - 1);
    }
}

void linspaceCuda(float* array, int n, float start, float end) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    linspaceKernel<<<gridSize, blockSize>>>(array, n, start, end);
    cudaDeviceSynchronize();
}

#endif