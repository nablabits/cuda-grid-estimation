/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H

void printArray(float *arr, int n=3) {
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

void checkArrays(float *gridX, float *gridY, float *gridZ)
{
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

__global__ void linspaceKernel(float *array, int size, float start, float end) {
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
  return exp(-0.5f * pow((x - mu) / sigma, 2.0f)) / (sigma * sqrt(2.0f * M_PI));
}

__global__ void normalPdfKernel(float *likes, float *gridX, float *gridY, float *gridZ, int gridSize) {

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

void computeLikesCuda(float *likes, float *gridX, float *gridY, float *gridZ, int gridSize) {
  dim3 threadsPerBlock(256);
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x);

  normalPdfKernel<<<numBlocks, threadsPerBlock>>>(likes, gridX, gridY, gridZ, gridSize);

  printArray(gridX, 10);
  printArray(gridY, 10);
  printArray(gridZ, 10);
  printArray(likes, 10);
}
#endif