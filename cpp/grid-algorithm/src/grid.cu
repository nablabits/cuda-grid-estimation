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
  const int rvs = 50;
  float *observations;
  /* Allocate space for results */
  CUDA_CALL(cudaMallocManaged(&observations, rvs * sizeof(float)));

  generateNormalCuda(rvs, mu, sigma, observations);

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

  const int vecSize = 101;
  const int gridSize = vecSize * vecSize * rvs;
  const float startMu = 18.0f;
  const float endMu = 22.0f;
  const float startSigma = 1.0f;
  const float endSigma = 3.0f;

  float *vectorMu, *vectorSigma, *likes;

  // TODO: these grid folks are auxiliary constructions that we might want to
  // put in a separate function
  float *gridX, *gridY, *gridZ;

  CUDA_CALL(cudaMallocManaged(&vectorMu, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&vectorSigma, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&likes, rvs * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gridX, gridSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gridY, gridSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gridZ, gridSize * sizeof(float)));

  CUDA_CALL(cudaMemset(likes, 0, rvs * sizeof(int)));

  linspaceCuda(vectorMu, vecSize, startMu, endMu);
  linspaceCuda(vectorSigma, vecSize, startSigma, endSigma);
  create3dGrid(
    vectorMu, vectorSigma, observations, gridX, gridY, gridZ, vecSize, rvs
  );

  checkArrays(gridX, gridY, gridZ);

  /*
  Compute the Likelihood Function
  */

  /* Cleanup */
  CUDA_CALL(cudaFree(observations));
  CUDA_CALL(cudaFree(vectorMu));
  CUDA_CALL(cudaFree(vectorSigma));
  CUDA_CALL(cudaFree(gridX));
  CUDA_CALL(cudaFree(gridY));
  CUDA_CALL(cudaFree(gridZ));
  CUDA_CALL(cudaFree(likes));

  return 0;
}