#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "../inc/cuda_functions.h"
#include "../inc/wrappers.h"

/*
How to run this test suite:

cd cpp/grid-algorithm
make all
./bin/tests
*/

// Test case for the CUDA function
TEST(CUDATest, GenerateRandomVariates) {
    unsigned int numElements = 3;
    curandState *devStates;
    float *devResults;
    float mu = 0.0f;
    float sigma = 1.0f;

    // Allocate Memory
    cudaMallocManaged(&devStates, numElements * sizeof(curandState));
    cudaMallocManaged(&devResults, numElements * sizeof(float));

    // Call the CUDA function
    generate_normal_kernel<<<1, 1>>>(
        devStates, numElements, mu, sigma, devResults
    );

    // Copy the result back to the host
    cudaDeviceSynchronize();

    // Perform assertions to check if the CUDA function worked as expected
    EXPECT_NEAR(devResults[0], 0.00459315, 1e-5);

    // Free memory
    cudaFree(devStates);
    cudaFree(devResults);
}


TEST(CUDATest, Linspaces) {
  // SetUp
  const int size = 3;
  const float start = 1.0f;
  const float end = 2.0f;

  float * vecX;
  cudaMallocManaged(&vecX, size * sizeof(float));

  // Act
  linspaceCuda(vecX, size, start, end);

  // Assert
  float expectedVecX[size] = {1.0f, 1.5f, 2.0f};
  ASSERT_TRUE(std::equal(vecX, vecX + size, expectedVecX));

  // Tear down
  cudaFree(vecX);
}


TEST(CUDATest, GenerateGrids) {
  // SetUp
  const int vecXYSize = 3;
  const int vecZSize = 2;
  const int gridSize = vecXYSize * vecXYSize * vecZSize;

  float *vecX, *vecY, *vecZ;
  float *gridX, *gridY, *gridZ;

  cudaMallocManaged(&vecX, vecXYSize * sizeof(float));
  cudaMallocManaged(&vecY, vecXYSize * sizeof(float));
  cudaMallocManaged(&vecZ, vecZSize * sizeof(float));
  cudaMallocManaged(&gridX, gridSize * sizeof(float));
  cudaMallocManaged(&gridY, gridSize * sizeof(float));
  cudaMallocManaged(&gridZ, gridSize * sizeof(float));

  // Act
  linspaceCuda(vecX, vecXYSize, 1, 3);
  linspaceCuda(vecY, vecXYSize, 1, 3);
  linspaceCuda(vecZ, vecZSize, 4, 5);
  create3dGridCuda(vecX, vecY, vecZ, gridX, gridY, gridZ, vecXYSize, vecZSize);

  // Assert
  float expectedX[18] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3};
  float expectedY[18] = {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3};
  float expectedZ[18] = {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5};

  ASSERT_TRUE(std::equal(gridX, gridX + gridSize, expectedX));
  ASSERT_TRUE(std::equal(gridY, gridY + gridSize, expectedY));
  ASSERT_TRUE(std::equal(gridZ, gridZ + gridSize, expectedZ));

  // TearDown
  cudaFree(vecX);
  cudaFree(vecY);
  cudaFree(vecZ);
  cudaFree(gridX);
  cudaFree(gridY);
  cudaFree(gridZ);
}


TEST(CUDATest, ComputeDensities) {
  int gridSize = 3;
  float *vecX, *vecY, *vecZ, *likes;

  cudaMallocManaged(&vecX, gridSize * sizeof(float));
  cudaMallocManaged(&vecY, gridSize * sizeof(float));
  cudaMallocManaged(&vecZ, gridSize * sizeof(float));
  cudaMallocManaged(&likes, gridSize * sizeof(float));

  linspaceCuda(vecX, gridSize, 20.0f, 22.0f);  // [20, 21, 22]
  linspaceCuda(vecY, gridSize, 1.0f, 3.0f);  // [1, 2, 3]
  linspaceCuda(vecZ, gridSize, 20.0f, 21.0f);  // [20, 20.5, 21]

  computeDensitiesCuda(likes, vecX, vecY, vecZ, gridSize);

  EXPECT_NEAR(likes[0], 0.39894228, 1e-5);
  EXPECT_NEAR(likes[1], 0.19333406, 1e-5);
  EXPECT_NEAR(likes[2], 0.12579441, 1e-5);

  // TearDown
  cudaFree(vecX);
  cudaFree(vecY);
  cudaFree(vecZ);
  cudaFree(likes);
}


TEST(CUDATest, ReshapeArray) {
  float *array = new float[10] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int rows = 2;
  int cols = 5;
  double **output = new double*[rows];
  for (int i = 0; i < rows; i++) {
    output[i] = new double[cols];
  }
  reshapeArray<float, double>(array, output, cols, rows);
  int expected[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      EXPECT_EQ(output[i][j], expected[j + i * cols]);
    }
  }

  // TearDown
  for (int i = 0; i < rows; i++) {
    delete[] output[i];
  }
  delete[] output;
  delete[] array;
}


TEST(CUDATest, ComputeLikes) {
  float *densities;
  double *likes;
  double **likesMatrix;

  int rows = 2;
  int cols = 5;

  cudaMallocManaged(&densities, rows * cols * sizeof(float));
  cudaMallocManaged(&likes, rows * sizeof(float));
  cudaMallocManaged(&likesMatrix, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    cudaMallocManaged(&likesMatrix[i], cols * sizeof(double));
  }

  linspaceCuda(densities, rows * cols, 1.0f, 10.0f);

  reshapeArray<float, double>(densities, likesMatrix, cols, rows);
  computeLikesCuda(likes, likesMatrix, rows, cols);

  EXPECT_EQ(likes[0], 120);  // 1*2*3*4*5
  EXPECT_EQ(likes[1], 30240);  // 6*7*8*9*10

  // TearDown
  for (int i = 0; i < rows; i++) {
    cudaFree(likesMatrix[i]);
  }
  cudaFree(likesMatrix);
  cudaFree(densities);
  cudaFree(likes);

}

TEST(CUDATest, ComputePosterior) {
  int size = 3;
  thrust::device_vector<double> likesV(size);
  thrust::device_vector<double> posteriorV(size);

  // Populate the likes vector
  thrust::sequence(likesV.begin(), likesV.end(), 1.0);  // 1.0, 2.0, 3.0

  computePosteriorCuda(likesV, posteriorV);

  EXPECT_NEAR(posteriorV[0], 0.166667, 1e-5);  // 1/6
  EXPECT_NEAR(posteriorV[1], 0.333333, 1e-5);  // 2/6
  EXPECT_EQ(posteriorV[2], 0.5);  // 3/6
}

TEST(CUDATest, ComputeExpectations) {
  /*
       4     5    6     p   w      E
  1   .1    .1   .1  | .3  .3   |
  2   .1    .2   .1  | .4  .8   |  2
  3   .1    .1   .1  | .3  .89  |
  --------------------
  p   .3    .4   .3
  w  1.2   2.0  1.8
  --------------------
  E        5.0
  */
  int size = 9;
  thrust::device_vector<double> posterior(size);
  thrust::fill(posterior.begin(), posterior.end(), .1);
  posterior[4] = .2;

  float *vectorMu, *vectorSigma;
  cudaMallocManaged(&vectorMu, 3 * sizeof(float));
  cudaMallocManaged(&vectorSigma, 3 * sizeof(float));

  linspaceCuda(vectorMu, 3, 1.0f, 3.0f);
  linspaceCuda(vectorSigma, 3, 4.0f, 6.0f);

  double* expectations = computeExpectationsWrapper(
    posterior, vectorMu, vectorSigma
  );

  EXPECT_EQ(expectations[0], 2.0f);
  EXPECT_EQ(expectations[1], 5.0f);


  cudaFree(vectorMu);
  cudaFree(vectorSigma);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}