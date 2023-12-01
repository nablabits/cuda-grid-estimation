// std library living in the compiler, see
// https://en.cppreference.com/w/cpp/standard_library
#include <iostream>
#include <math.h>  // C std library
#include <cuda_runtime.h>

#include "../inc/add.h"

// function to add the elements of two arrays element wise cuda flavour. The
// reserved identifier `__global__` is exclusive of CUDA and not part of the
// std.

int main(void)
{
  // we shift the binary repr of `1` 20 positions to the left, which makes the
  // number 2^20=1,048,576 (this also works in Python and Julia)
  int N = 1<<20;

  // Assign dynamically memory addresses on the Unified Memory in CUDA, in a
  // way that can be edited in place by `add`
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Now that we have the memory allocated on the Unified Memory, we can
  // populate it, the first array will contain 1s and the second 2s.
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU. The `<<<` is not part of c++.
  // threads are multiples of 32 (a warp)
  int blockSize = 256;  // the number of threads per block
  int numBlocks = (N + blockSize - 1) / blockSize;  // efectivelly, 4096 blocks
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}