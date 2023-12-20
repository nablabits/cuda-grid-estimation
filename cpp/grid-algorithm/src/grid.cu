#include <iostream>

#include "../inc/cuda_functions.h"


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

/********************
* Wrapper Functions *
********************/

void computeDensitiesWrapper(float *vecX, float *vecY, float *vecZ, float *output,
                         int startX, int endX, int startY, int endY,
                         int vecXYSize, int vecZSize)
{
  /*
  Wrap the operations needed to compute the densities of the likelihood function.
  */

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

  create3dGridCuda(
    vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize
  );

  checkArrays(gridX, gridY, gridZ);

  computeDensitiesCuda(output, gridX, gridY, gridZ, gridSize);

  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}

void computeLikesWrapper(float *densities, double *likes, int densitiesSize, int likesSize)
{
  /*
  Wrap the operations needed to compute the likelihood function.
  */

  double **likesMatrix;
  int rows = likesSize;  // 101x101 rows
  int cols = densitiesSize / likesSize;  // of 50 elements each

  if (densitiesSize != rows * cols) {
    printf("ERROR: likesSize != rows * cols\n");
    return;
  }

  cudaMallocManaged(&likesMatrix, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    cudaMallocManaged(&likesMatrix[i], cols * sizeof(double));
  }

  reshapeArray(densities, likesMatrix, cols, rows);
  computeLikesCuda(likes, likesMatrix, rows, cols);

  for (int i = 0; i < cols; i++) {
    cudaFree(likesMatrix[i]);
  }
  cudaFree(likesMatrix);
}


int main(void)
{
  /*******************************
  * Generate the random variates *
  *******************************/

  /*
  We start by generating the random variates that will serve as a proxy of some
  process in the real world of evidence gathering.
  */

  /* These are the hidden folks we want to estimate*/
  const float mu = 20.0f;
  const float sigma = 2.0f;

  const int rvsSize = 50;
  float *observations;

  CUDA_CALL(cudaMallocManaged(&observations, rvsSize * sizeof(float)));

  generateNormalCuda(rvsSize, mu, sigma, observations);

  // Due to seed, the first element should be 18.5689, let's make sure we are
  // we are close enough from it.
  if (observations[0] - 18.5689f > 0.0001f) {
    std::cout << "Oh noh! unexpected observations" << observations[0];
    std::cout << std::endl;
    return 1;
  }

  printArray(observations, rvsSize);

  /*******************
  * Create the grids *
  *******************/

 /*
  We need a grid with the outer product of three vectors that will represent the
  combinations of the parameters we want to estimate. This is, if our vectors
  are [1, 2, 3] & [4, 5 ,6] & [1, 2] then our outer product will be:

  [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
  [4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]
  [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

  We use extensively the word `grid`, but it won't be a proper grid as the
  output will be a vector. Granted, a vector with the same elements as the grid,
  v1 x v2 x v3, but a vector after all.
 */

  const int vecSize = 101;
  const int gridSize = vecSize * vecSize * rvsSize;
  const float startMu = 18.0f;
  const float endMu = 22.0f;
  const float startSigma = 1.0f;
  const float endSigma = 3.0f;

  float *vectorMu, *vectorSigma, *densities;

  CUDA_CALL(cudaMallocManaged(&vectorMu, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&vectorSigma, vecSize * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&densities, gridSize * sizeof(float)));
  CUDA_CALL(cudaMemset(densities, 0, gridSize * sizeof(int)));

  /*************************
  * Compute the Likelihood *
  *************************/

  /*
  Computing the likelihood involves two steps: First we compute the densities
  over the observations for each pair of mu, sigma. Then, we take the product
  over those densities as they can be thought as a joint probability.

  mus       [   1,    1,    1,    1, ...]
  sigmas    [   4,    4,    5,    5, ...]
  obs       [   1,    2,    1,    2, ...]
  densities [.099, .096, .079, .078, ...]
  likes     [  0.0096,      .0062,   ...]
  */

  computeDensitiesWrapper(vectorMu, vectorSigma, observations, densities,
                          startMu, endMu, startSigma, endSigma, vecSize,
                          rvsSize);

  double *likes;
  const int likesSize = vecSize * vecSize;
  CUDA_CALL(cudaMallocManaged(&likes, likesSize * sizeof(double)));
  computeLikesWrapper(densities, likes, gridSize, likesSize);


  /**********
  * Cleanup *
  **********/

  CUDA_CALL(cudaFree(observations));
  CUDA_CALL(cudaFree(vectorMu));
  CUDA_CALL(cudaFree(vectorSigma));
  CUDA_CALL(cudaFree(densities));
  CUDA_CALL(cudaFree(likes));

  return 0;
}