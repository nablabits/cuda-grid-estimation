#include <iostream>
#include <curand_kernel.h>

#include "../inc/normal_kernel.h"
#include "../inc/arrays.h"


// this is a macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

/*
Sources:
https://developer.nvidia.com/blog/efficient-cuda-debugging-memory-initialization-and-thread-synchronization-with-nvidia-compute-sanitizer/

https://docs.nvidia.com/cuda/curand/group__DEVICE.html

https://chat.openai.com/c/ba212caf-491f-4dfd-ac4f-ce2132672561

TODO: Move them to a note on obsidian once we are done along with the learnings

How to run this:

cd cpp/grid-algorithm
make all

# always check that we are not doing something spooky under the hood
compute-sanintizer ./bin/grid

# Run the binary
./bin/grid
*/


int main(void)
{
  /* These are the hidden folks we want to estimate*/
  const float mu = 20.0f;
  const float sigma = 2.0f;

  /* Generate the random variates */
  /********************************/
  unsigned int numElements = 50;
  float *observations;
  /* Allocate space for results */
  CUDA_CALL(cudaMallocManaged(&observations, numElements * sizeof(float)));

  generateNormalCuda(numElements, mu, sigma, observations);

  // Due to seed, the first element should be 18.5689, let's check how close
  // we are from it
  if (observations[0] - 18.5689f > 0.0001f) {
    std::cout << "Oh noh! " << observations[0] << std::endl;
    return 1;
  }

  /*
  Create the grids
  ****************
  */
  float *vectorMu;
  float *vectorSigma;
  const int size = 101;
  const int gridSize = size * size;
  const float startMu = 18.0f;
  const float endMu = 22.0f;
  const float startSigma = 1.0f;
  const float endSigma = 3.0f;

  float *gridX;
  float *gridY;

  CUDA_CALL(cudaMallocManaged(&vectorMu, size * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&vectorSigma, size * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gridX, gridSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gridY, gridSize * sizeof(float)));

  linspaceCuda(vectorMu, size, startMu, endMu);
  linspaceCuda(vectorSigma, size, startSigma, endSigma);
  createGridCuda(vectorMu, vectorSigma, gridX, gridY, size);

  // A couple of sanity checks
  if (gridX[5050] != 20.0f) {
    std::cout << "Oh noh! " << gridX[5050] << std::endl;
    return 1;
  }

  // 101 vectorMu * 1/2 vectorSigma; 5050 index
  if (gridY[151] != 2.0f) {
    std::cout << "Oh noh! " << gridY[151] << std::endl;
    return 1;
  }

  /* It may be possible to use thrust::reduce to take the product over axis*/


  /* Cleanup */
  CUDA_CALL(cudaFree(observations));
  CUDA_CALL(cudaFree(vectorMu));
  CUDA_CALL(cudaFree(vectorSigma));
  CUDA_CALL(cudaFree(gridX));
  CUDA_CALL(cudaFree(gridY));

  return 0;
}