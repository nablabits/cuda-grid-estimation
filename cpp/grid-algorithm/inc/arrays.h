/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H

void printArray(float *arr, int n=3) {
  /* Handy function to print arrays. */
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

void checkArrays(float *gridX, float *gridY, float *gridZ)
{
  /* Run a sanity check on the grids to ensure they hold during development. */

  // 50 * 101 * 50
  // One full cycle of observations times one full cycle of sigmas times half
  // cycle of mus
  int idx = 252500;
  if (gridX[idx] != 20.0f) {
    std::cout << "Oh noh! GridX right mismatch: " << gridX[idx] << std::endl;
  }
  if (gridX[idx - 1] != 19.96f) {
    std::cout << "Oh noh! GridX left mismatch: " << gridX[idx - 1] << std::endl;
  }

  idx = 50;  // One full cycle of observations
  if (gridY[idx] != 1.02f) {
    std::cout << "Oh noh! GridY right mismatch" << gridY[idx] << std::endl;
  }
  if (gridY[idx - 1] != 1.0f) {
    std::cout << "Oh noh! GridY left mismatch" << gridY[idx - 1] << std::endl;
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

void linspaceCuda(float* array, int size, float start, float end) {
    /*
    Define the configuration of the linspace kernel.

    array: output array
    n: length of array
    start: start of range (included)
    end: end of range (included)
    */
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    linspaceKernel<<<numBlocks, threadsPerBlock>>>(array, size, start, end);
    cudaDeviceSynchronize();
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

void create3dGrid(float *vecX, float *vecY, float *vecZ,
                  float *gridX, float *gridY, float *gridZ,
                  int vecXYSize, int vecZSize)
{
  /* Define the configuration of the create3dGrid kernel. */
  dim3 threadsPerBlock(4, 4, 4);
  dim3 numBlocks((vecXYSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (vecXYSize + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (vecZSize + threadsPerBlock.z - 1) / threadsPerBlock.z);

  create3dGridKernel<<<numBlocks, threadsPerBlock>>>(
    vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize
  );

  cudaDeviceSynchronize();
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

__global__ void computeLikesKernel(float *likes, float *gridX, float *gridY,
                                   float *gridZ, int gridSize)
{
  /* Compute the likelihood function on the device. */

  // at this point we have three vectors that represent our 3d grid, so we just
  // need to iterate over the arrays.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < gridSize) {
    float x = gridZ[i];
    float sigma = gridY[i];
    float mu = gridX[i];
    likes[i] = normalPdf(x, mu, sigma);
  }
}

void computeLikesCuda(float *likes, float *gridX, float *gridY, float *gridZ,
                      int gridSize)
{
  /* Define the configuration of the computeLikesKernel kernel. */
  dim3 threadsPerBlock(256);
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  computeLikesKernel<<<numBlocks, threadsPerBlock>>>(likes, gridX, gridY, gridZ,
                                                     gridSize);

  cudaDeviceSynchronize();

  printArray(gridX, 10);
  printArray(gridY, 10);
  printArray(gridZ, 10);
  printArray(likes, 10);
}

void computeLikesWrapper(float *vecX, float *vecY, float *vecZ, float *output,
                         int startX, int endX, int startY, int endY,
                         int vecXYSize, int vecZSize)
{
  /* Wrap the operations needed to compute the likelihood function. */

  /* It seems that we are reinventing the wheel a bit as we could use the
  cuTENSOR library. This, however, has a steeeeep learning curve 😕
  */

  float *gridX, *gridY, *gridZ;
  int gridSize = vecXYSize * vecXYSize * vecZSize;
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&gridZ, gridSize * sizeof(float));

  linspaceCuda(vecX, vecXYSize, startX, endX);
  linspaceCuda(vecY, vecXYSize, startY, endY);

  create3dGrid(
    vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize
  );

  checkArrays(gridX, gridY, gridZ);

  computeLikesCuda(output, gridX, gridY, gridZ, gridSize);

  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}

#endif