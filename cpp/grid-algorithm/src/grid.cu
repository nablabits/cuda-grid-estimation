#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include "../inc/normal_kernel.h"


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
  const unsigned int threadsPerBlock = 64;
  const unsigned int blockCount = 64;
  const unsigned int totalThreads = threadsPerBlock * blockCount;  // 4096

  /* These are the hidden folks we want to estimate*/
  const float mu = 20.0f;
  const float sigma = 5.0f;

  /* Generate the random variates */
  /********************************/
  unsigned int numElements = 50;
  curandState *devStates;
  float *devResults;

  /* MEMORY ALLOCATION */
  /* Allocate space for prng states */
  CUDA_CALL(cudaMallocManaged(&devStates, totalThreads *sizeof(curandState)));

  /* Allocate space for results */
  CUDA_CALL(cudaMallocManaged(&devResults, totalThreads * sizeof(float)));

  setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);

  generate_normal_kernel<<<blockCount, threadsPerBlock>>>(
    devStates, numElements, mu, sigma, devResults
  );

  cudaDeviceSynchronize();

  // Due to seed, the first element should be 16.4222, let's check how close
  // we are from it
  if (devResults[0] - 16.4222f > 0.0001f) {
    std::cout << "Oh noh! " << devResults[0] << std::endl;
  }

  /* Cleanup */
  CUDA_CALL(cudaFree(devStates));
  CUDA_CALL(cudaFree(devResults));

  return 0;
}