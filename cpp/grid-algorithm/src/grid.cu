#include <iostream>
#include <math.h>  // C std library
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
TODO: I will keep working in the test-sandbox branch to have at hand the 
findings in `simple-approach` & `tests`. Once I have some test working for this
I can move back to feature/grid-algorithm
*/


int main(void)
{
  const unsigned int threadsPerBlock = 64;
  const unsigned int blockCount = 64;
  const unsigned int totalThreads = threadsPerBlock * blockCount;  // 4096

  unsigned int numElements = 50;
  curandState *devStates;
  float *devResults, *hostRVs;


  /* MEMORY ALLOCATION */
  // TODO: Find out the way to write some tests
  // TODO: change this to cudaMallocManaged once is working

  /* Allocate space for results on host */
  hostRVs = (float *)calloc(totalThreads, sizeof(float));

  /* Allocate space for prng states on device */
  CUDA_CALL(
    cudaMalloc(&devStates, totalThreads *sizeof(curandState))
  );

  /* Allocate space for results on device */
  CUDA_CALL(cudaMalloc(&devResults, totalThreads *
            sizeof(float)));


  /* Set results to 0 */
  CUDA_CALL(cudaMemset(devResults, 0, totalThreads *
            sizeof(float)));

  setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);

  generate_normal_kernel<<<blockCount, threadsPerBlock>>>(
    devStates, numElements, devResults
  );

  cudaDeviceSynchronize();

  /* Copy device memory to host */
  CUDA_CALL(cudaMemcpy(hostRVs, devResults, totalThreads *
      sizeof(float), cudaMemcpyDeviceToHost));

  unsigned int count = 0;
  unsigned int withinOneSD = 0;
  for (int i = 0; i < numElements; i++) {
    std::cout << hostRVs[i] << std::endl;
    if (hostRVs != 0)
      count++;
    if (hostRVs[i] > -1.0 && hostRVs[i] < 1.0) {
      withinOneSD++;
    }
  }

  std::cout << "RVs generated: " << count << std::endl;
  std::cout << "Within one SD: " << (float)withinOneSD / count << std::endl;

  /* Cleanup */
  CUDA_CALL(cudaFree(devStates));
  CUDA_CALL(cudaFree(devResults));
  free(hostRVs);

  return 0;
}