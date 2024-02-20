#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "../inc/cuda_functions.h"
#include "../inc/kernels.h"
#include "../inc/wrappers.h"


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
  /*******************************
  * Generate the Random Variates *
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
  * Create the Grids *
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
  const float startMu = mu - 2;
  const float endMu = mu + 2;
  const float startSigma = sigma - 1;
  const float endSigma = sigma + 1;

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

  mus       [   1,    1,    1,    1, ...]  101
  sigmas    [   4,    4,    5,    5, ...]  101
  obs       [   1,    2,    1,    2, ...]  50
  densities [.099, .096, .079, .078, ...]  101x101x50
  likes     [  0.0096,      .0062,   ...]  101x101
  */

  computeDensitiesWrapper(vectorMu, vectorSigma, observations, densities,
                          startMu, endMu, startSigma, endSigma, vecSize,
                          rvsSize);

  double *likes;
  const int likesSize = vecSize * vecSize;
  CUDA_CALL(cudaMallocManaged(&likes, likesSize * sizeof(double)));
  computeLikesWrapper(densities, likes, gridSize, likesSize);

  /*************************
   * Compute the Posterior *
  *************************/

  /*
  In principle we will asume a flat prior, which has no impact on the
  likelihoods. But we still need to normalize them so they will add up to 1.
  */


  // We start by building thrust vectors out of the likes array so we can
  // easily and efficiently compute the sum of the array. The first bit is
  // taking the initial value of `likes` and then copying over the rest of the
  // array up to `likesSize` with `likes + likesSize`
  // Then, we just create another vector that will hold the posteriors.

  // TODO: it might be a good idea to use host vectors before and after
  thrust::device_vector<double> likesV(likes, likes + likesSize);
  thrust::device_vector<double> posteriorV(likesSize);
  computePosteriorCuda(likesV, posteriorV, likesSize);

  /*************************
   * Compute the Marginals *
  *************************/

  /*
  Now that we have the posterior we can compute the marginals and with them, the
  expectations for the parameters that hopefully will land closer to the values
  we set to generate the variates.
  */
  double* expectations = computeExpectationsWrapper(
    posteriorV, vectorMu, vectorSigma
  );

  std::cout << "Inferred mu: " << expectations[0]
  << "; Actual mu: " << mu <<  std::endl;
  std::cout << "Inferred sigma: " << expectations[1]
  << "; Actual sigma: " << sigma << std::endl;


  /**********
  * Cleanup *
  **********/

  // TODO: maybe we can free the elements as soon as we dont need them.
  CUDA_CALL(cudaFree(observations));
  CUDA_CALL(cudaFree(vectorMu));
  CUDA_CALL(cudaFree(vectorSigma));
  CUDA_CALL(cudaFree(densities));
  CUDA_CALL(cudaFree(likes));

  return 0;
}