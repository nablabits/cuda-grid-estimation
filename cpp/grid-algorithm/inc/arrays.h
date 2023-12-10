/* Handy function to create arrays */
#include <cuda_runtime.h>

#ifndef ARRAYS_H
#define ARRAYS_H


__global__ void linspaceKernel(float *array, int size, float start, float end) {
  /*
  Create a linearly spaced array on the device.

  array: output array
  n: length of array
  start: start of range
  end: end of range
  */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = start + idx * (end - start) / (size - 1);
    }
}

void linspaceCuda(float* array, int size, float start, float end) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    linspaceKernel<<<gridSize, blockSize>>>(array, size, start, end);
    cudaDeviceSynchronize();
}


__global__ void createGridKernel(
  int *vectorX, int *vectorY, int *gridX, int *gridY, int size
  )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ii = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("i: %d, ii: %d\n", i, ii);
  // int vx[3] = {0, 1, 2};
  // int vy[3] = {0, 1, 2};
  // for (int h = 0; h < 9; h++) {
  //   printf("vector at %d: %d\n", h, vectorX[h]);
  // }


  int pos = i * 3 + ii;
  if (i < 3 && ii < 3) {
    gridX[pos] = vectorX[i];
    gridY[pos] = vectorY[ii];
  }
}


void simpleCreateGridCuda()
{
  int vecX_[3] = {0, 1, 2};
  int vecY_[3] = {0, 1, 2};

  int *vecX, *vecY;
  cudaMallocManaged(&vecX, 3 * sizeof(int));
  cudaMallocManaged(&vecY, 3 * sizeof(int));

  for (int i = 0; i < 3; i++) {
    vecX[i] = vecX_[i];
    vecY[i] = vecY_[i];
  }



  int size = 9;
  int *gX, *gY;
  cudaMallocManaged(&gX, size * sizeof(int));
  cudaMallocManaged(&gY, size * sizeof(int));


  dim3 threadsPerBlock(9, 9);  // You can adjust the block size as needed
  dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  printf("threadsPerBlock.x: %d, threadsPerBlock.y: %d\n", threadsPerBlock.x, threadsPerBlock.y);
  printf("numBlocks.x: %d, numBlocks.y: %d\n", numBlocks.x, numBlocks.y);

  printf("%d\n", vecX[2]);

  createGridKernel<<<numBlocks, threadsPerBlock>>>(vecX, vecY, gX, gY, size);

  cudaDeviceSynchronize();

  for (int i = 0; i < 9; i++) {
    std::cout << gX[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 9; i++) {
    std::cout << gY[i] << ", ";
  }
  std::cout << std::endl;

  cudaFree(gX);
  cudaFree(gY);
}






// TODO: this should be a device function
void createGrid(
  float *vectorX, float *vectorY, float *gridX, float *gridY, int size,
  float startMu, float endMu, float startSigma, float endSigma
  )
{
  linspaceCuda(vectorX, size, startMu, endMu);
  linspaceCuda(vectorY, size, startSigma, endSigma);
  int pos = 0;
  for (int i = 0; i < size; i++) {
    for (int ii = 0; ii < size; ii++) {
        gridX[pos] = vectorX[ii];
        gridY[pos] = vectorY[i];
        pos++;
    }
  }
}

#endif