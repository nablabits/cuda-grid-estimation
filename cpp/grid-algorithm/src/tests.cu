#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../inc/normal_kernel.h"
#include "../inc/arrays.h"

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

TEST(CUDATest, ComputeLikes) {
  int gridSize = 3;
  float *vecX, *vecY, *vecZ, *likes;

  cudaMallocManaged(&vecX, gridSize * sizeof(float));
  cudaMallocManaged(&vecY, gridSize * sizeof(float));
  cudaMallocManaged(&vecZ, gridSize * sizeof(float));
  cudaMallocManaged(&likes, gridSize * sizeof(float));

  linspaceCuda(vecX, gridSize, 20.0f, 22.0f);  // [20, 21, 22]
  linspaceCuda(vecY, gridSize, 1.0f, 3.0f);  // [1, 2, 3]
  linspaceCuda(vecZ, gridSize, 20.0f, 21.0f);  // [20, 20.5, 21]

  computeLikesCuda(likes, vecX, vecY, vecZ, gridSize);

  EXPECT_NEAR(likes[0], 0.39894228, 1e-5);
  EXPECT_NEAR(likes[1], 0.19333406, 1e-5);
  EXPECT_NEAR(likes[2], 0.12579441, 1e-5);

  // TearDown
  cudaFree(vecX);
  cudaFree(vecY);
  cudaFree(vecZ);
  cudaFree(likes);
}




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}