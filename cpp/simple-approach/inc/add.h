#include <cuda_runtime.h>

#ifndef ADD_H
#define ADD_H

__global__
void add(int n, float *x, float *y)
{
  //This line calculates the unique index for each thread based on its block
  //index (blockIdx.x), thread index within the block (threadIdx.x), and block
  //size (blockDim.x)
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // The stride variable represents the total number of threads in the grid
  // (blockDim.x * gridDim.x). It defines the step size between elements that
  // each thread will process.
  int stride = blockDim.x * gridDim.x;

  // Run the loop
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

#endif