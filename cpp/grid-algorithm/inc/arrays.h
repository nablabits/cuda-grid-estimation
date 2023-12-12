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


__global__ void createGridKernel(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size
  )
{
  /*
  Create the outer product of two verctors on the device.

  We need a grid with the outer product of two vectors that will represent the
  combinations of the parameters we want to estimate. This is, if our vectors
  are [1, 2, 3] & [4, 5 ,6], then our outer product will be:

  [1, 2, 3, 1, 2, 3, 1, 2, 3]
  [4, 4, 4, 5, 5, 5, 6, 6, 6]

  TODO: we treat vectors and grids as separate objects. A possible improvement
  here could be to treat them as a single multidimensional array.

  Arguments:
    vectorX: the first vector
    vectorY: the second vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    size: the size of the vectors
  */

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;

  int pos = i * size + ii;
  if (i < size && ii < size) {
    gridX[pos] = vectorX[i];
    gridY[pos] = vectorY[ii];
  }
}

void createGridCuda(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size
  )
{
  /*
  Define the configuration of createGrid kernel and call it.

  Arguments:
    vectorX: the first vector
    vectorY: the second vector
    gridX: the grid of the first vector to be filled
    gridY: the grid of the second vector to be filled
    size: the size of the vectors
  */

  dim3 threadsPerBlock(16, 16);  // You can adjust the block size as needed
  dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  createGridKernel<<<numBlocks, threadsPerBlock>>>(
    vectorX, vectorY, gridX, gridY, size
  );

  cudaDeviceSynchronize();
}

__device__ float normalPdf(float x, float mu, float sigma) {
  return exp(-0.5f * pow((x - mu) / sigma, 2.0f)) / (sigma * sqrt(2.0f * M_PI));
}

__global__ void normalPdfKernel(float *likes, float *gridX, float *gridY, int gridSize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < gridSize) {
    float x = 20.5f;
    float sigma = gridY[i];
    float mu = gridX[i];
    likes[i] = normalPdf(x, mu, sigma);
// TODO: Continue here, extend this function to compute the grids for the
// real values.
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

void create3dGrid() {
  int paramSize = 3;
  int obsSize = 2;
  int gridSize = paramSize * paramSize * obsSize;

  float *vecX, *vecY, *vecZ;
  float *gridX, *gridY, *gridZ;
  cudaMallocManaged(&vecX, paramSize * sizeof(float));
  cudaMallocManaged(&vecY, paramSize * sizeof(float));
  cudaMallocManaged(&vecZ, obsSize * sizeof(float));
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&gridZ, gridSize * sizeof(float));

  linspaceCuda(vecX, paramSize, 0.0f, 2.0f);
  linspaceCuda(vecY, paramSize, 0.0f, 2.0f);
  linspaceCuda(vecZ, obsSize, 0.0f, 1.0f);

  dim3 threadsPerBlock(4, 4, 4);
  dim3 numBlocks((paramSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (paramSize + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

  create3dGridKernel<<<numBlocks, threadsPerBlock>>>(
    vecX, vecY, vecZ, gridX, gridY, gridZ, paramSize, obsSize
  );

  cudaDeviceSynchronize();

  printArray(gridX, gridSize);
  printArray(gridY, gridSize);
  printArray(gridZ, gridSize);

  cudaFree(vecX);
  cudaFree(vecY);
  cudaFree(vecZ);
  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}


void simpleLikelihood() {
  /*
  Next steps:
  - Extend the function to take a vector of observations
  */
  // float observations = 20.5f;
  int vecSize = 3;
  int gridSize = vecSize * vecSize;

  float *vecX, *vecY;
  float *gridX, *gridY, *likes;
  cudaMallocManaged(&vecX, vecSize * sizeof(float));
  cudaMallocManaged(&vecY, vecSize * sizeof(float));
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&likes, gridSize * sizeof(float));

  linspaceCuda(vecX, 3, 19.0f, 21.0f);
  linspaceCuda(vecY, 3, 1.0f, 3.0f);
  createGridCuda(vecX, vecY, gridX, gridY, vecSize);

  dim3 threadsPerBlock(16, 16);  // You can adjust the block size as needed
  dim3 numBlocks((gridSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (gridSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
  normalPdfKernel<<<numBlocks, threadsPerBlock>>>(likes, gridX, gridY, gridSize);

  printArray(gridX, gridSize);
  printArray(gridY, gridSize);
  printArray(likes, gridSize);

  // Clean up
  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(likes);
  cudaFree(vecX);
  cudaFree(vecY);
}

#endif