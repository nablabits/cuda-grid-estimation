#include <cuda_runtime.h>

#ifndef NORMAL_KERNEL_H
#define NORMAL_KERNEL_H

__global__ void setup_kernel(curandState *state)
{
  /*
  In CUDA programming, each thread has its own RNG state, which determines the
  sequence of random numbers generated by that thread. By initializing the RNG 
  state, we ensure that each thread starts with a distinct and independent 
  sequence of random numbers.
  */
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(42, id, 0, &state[id]);
}

__global__ void generate_normal_kernel(curandState *state,
                                unsigned int n,
                                float *result)
{
    /* 
    In this function we will be generating radom variates with curand_normal
    using the states defined in `setup_kernel`.
    */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // TODO: these should be args for the function
    float mu = 0.0f;
    float sigma = 1.0f;
    float rv;
    
    /* Generate pseudo-random normals */
    for(int i = index; i < n; i += stride) {
      /* Copy state to local memory for efficiency */
      curandState localState = state[index];

      rv = curand_normal(&localState);
      result[i] = mu + sigma * rv;

      /* Copy state back to global memory */
      state[i] = localState;
    }
}

#endif