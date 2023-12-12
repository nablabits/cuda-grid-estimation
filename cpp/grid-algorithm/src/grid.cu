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
  /*******************************
  * Generate the random variates *
  *******************************/

  /*
  We start by generating the random variates that will serve as a proxy of some
  process in the real world of evidence gathering.
  */

  /* These are the hidden folks we want to estimate*/
  const float mu = 20.0f;
  const float sigma = 2.0f;

  const int rvsSize = 50;
  float *observations;

  CUDA_CALL(cudaMallocManaged(&observations, rvsSize * sizeof(float)));

  generateNormalCuda(rvsSize, mu, sigma, observations);

  // Due to seed, the first element should be 18.5689, let's make sure we are
  // we are close enough from it.
  if (observations[0] - 18.5689f > 0.0001f) {
    std::cout << "Oh noh! unexpected observations" << observations[0];
    std::cout << std::endl;
    return 1;
  }

  /*******************
  * Create the grids *
  *******************/

 /*
  We need a grid with the outer product of three vectors that will represent the
  combinations of the parameters we want to estimate. This is, if our vectors
  are [1, 2, 3] & [4, 5 ,6] & [1, 2] then our outer product will be:

  [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
  [4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]
  [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

  We use extensively the word `grid`, but it won't be a proper grid as the
  output will be a vector. Granted, a vector with the same elements as the grid,
  v1 x v2 x v3, but a vector after all.
 */

  const int vecSize = 101;
  const int gridSize = vecSize * vecSize * rvsSize;
  const float startMu = 18.0f;
  const float endMu = 22.0f;
  const float startSigma = 1.0f;
  const float endSigma = 3.0f;

  float *vectorMu, *vectorSigma, *likes;

  CUDA_CALL(cudaMallocManaged(&vectorMu, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&vectorSigma, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&likes, gridSize * sizeof(float)));
  CUDA_CALL(cudaMemset(likes, 0, gridSize * sizeof(int)));

  computeLikesWrapper(vectorMu, vectorSigma, observations, likes,
                      startMu, endMu, startSigma, endSigma, vecSize, rvsSize);

  // TODO: continue here, write some tests

  /**********
  * Cleanup *
  **********/

  CUDA_CALL(cudaFree(observations));
  CUDA_CALL(cudaFree(vectorMu));
  CUDA_CALL(cudaFree(vectorSigma));
  CUDA_CALL(cudaFree(likes));

  return 0;
}