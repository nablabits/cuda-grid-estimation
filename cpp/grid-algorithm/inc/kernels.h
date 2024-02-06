#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef KERNELS_H
#define KERNELS_H

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
                                float mu,
                                float sigma,
                                float *result)
{
    /*
    In this function we will be generating radom variates with curand_normal
    using the states defined in `setup_kernel`.
    */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float rv;

    /* Generate pseudo-random normals */
    for(int i = index; i < n; i += stride) {
      /* Copy state to local memory for efficiency */
      curandState localState = state[index];

      rv = curand_normal(&localState);
      result[i] = mu + sigma * rv;

      /* Copy state back to global memory */
      state[i] = localState;  // TODO: this might not be necessary
    }
}

__global__ void linspaceKernel(float *array, int size, float start, float end)
{
  /*
  Create a linearly spaced array on the device.

  array: output array
  n: length of array
  start: start of range (included)
  end: end of range (included)
  */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = start + idx * (end - start) / (size - 1);
    }
}


__global__ void create3dGridKernel(float *vecX, float *vecY, float *vecZ,
                                   float *gridX, float *gridY, float *gridZ,
                                   int vecXYSize, int vecZSize)
{
  /*
  Create the outer product of three vectors on the device.

  For the purpose of this project we assume that both XY vectors are the same
  size and only Z differs.

  TODO: we treat vectors and grids as separate objects. A possible improvement
  here could be to treat them as a single multidimensional array.

  Arguments:
    vecX: the first vector
    vecY: the second vector
    vecZ: the third vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    vecXYSize: the size of the vectors vecX and vecY
    vecZSize: the size of the vector vecZ
  */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;
  int iii = blockIdx.z * blockDim.z + threadIdx.z;

  int idx = i * vecXYSize * vecZSize + ii * vecZSize + iii;

  if (i < vecXYSize && ii < vecXYSize && iii < vecZSize) {
    gridX[idx] = vecX[i];
    gridY[idx] = vecY[ii];
    gridZ[idx] = vecZ[iii];
  }
}


__device__ float normalPdf(float x, float mu, float sigma) {
  /*
  Calculate the PDF of a normal distribution on the device at some location x.

  Arguments:
    x: value
    mu: mean
    sigma: standard deviation
  */
  return exp(-0.5f * pow((x - mu) / sigma, 2.0f)) / (sigma * sqrt(2.0f * M_PI));
}

__global__ void computeDensitiesKernel(float *likes, float *gridX, float *gridY,
                                   float *gridZ, int gridSize)
{
  /* Compute the densities on the device.

  At this point we have three vectors that represent our 3d grid, so we just
  need to iterate over the arrays to get the densities for each pair of mu,
  sigma, over the observations (x).
  */


  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < gridSize) {
    float x = gridZ[i];
    float sigma = gridY[i];
    float mu = gridX[i];
    likes[i] = normalPdf(x, mu, sigma);
  }
}


__global__ void computeLikesKernel(double *likes, double **likesMatrix, int rows, int cols)
{
  /* Compute the likelihoods function on the device.

  1   2   3    4    5    6    7    8    9    10
  |___|   |____|    |____|    |____|    |_____|
    |        |        |          |         |
    2       12        30        56        90
    |________|        |__________|         |
         |                   |             |
         24               1680             |
         |___________________|             |
                  |                        |
                  40320                   90
                  |________________________|
                               |
                            3628800
  The simplest example is: in each iteration, group the elements in the array in
  pairs and compute their product so the max number of iterations will be
  log2(N).

  However, the final algorithm is a bit more elaborated as we are passing a
  matrix of 10201x50 elements (101*101x50) and the product is computed over rows
  which are the densities for each of the 50 observations.

       row 0              row 1
  --------------    ----------------
  1   2   3    4    5    6    7    8
  |___|   |____|    |____|    |____|
    |        |        |          |
    2       12        30        56
    |________|        |__________|
        |                   |
        24               1680
      output[0]         output[1]
  */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Use a parallel reduction to calculate the product
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (idx < stride && idx + stride < cols) {
      for (int i = 0; i < rows; i++) {
        likesMatrix[i][idx] *= likesMatrix[i][idx + stride];
      }
    }
    __syncthreads();
  }

  // Store the result in the outcome
  if (idx == 0) {
    for (int i = 0; i < rows; i++) {
      likes[i] = likesMatrix[i][0];
    }
  }
}


#endif