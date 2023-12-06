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
*/


int main(void)
{
  const unsigned int threadsPerBlock = 64;
  const unsigned int blockCount = 64;

  unsigned int numElements = 50;
  curandState *devStates;
  float *devResults;

  /* MEMORY ALLOCATION */
  /* Allocate space for prng states */
  CUDA_CALL(cudaMallocManaged(&devStates, numElements *sizeof(curandState)));

  /* Allocate space for results */
  CUDA_CALL(cudaMallocManaged(&devResults, numElements * sizeof(float)));

  setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);

  generate_normal_kernel<<<blockCount, threadsPerBlock>>>(
    devStates, numElements, devResults
  );

  cudaDeviceSynchronize();

  // Due to seed, the first element should be -0.715557, let's check how close
  // we are from it
  if (devResults[0] + 0.715557f > 0.00001f) {
    std::cout << "Oh noh! " << devResults[0] << std::endl;
  }

  /* Cleanup */
  CUDA_CALL(cudaFree(devStates));
  CUDA_CALL(cudaFree(devResults));

  return 0;
}