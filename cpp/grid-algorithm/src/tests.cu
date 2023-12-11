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


class CUDATest : public ::testing::Test {
protected:
    // Set up any common resources or configurations needed for the tests
    // For CUDA, you might allocate device memory, etc.
    void SetUp() override {
        // Add setup code here if needed
    }

    // Clean up any resources allocated in SetUp
    void TearDown() override {
        // Add cleanup code here if needed
    }
};

// Test case for the CUDA function
TEST_F(CUDATest, GenerateRandomVariates) {
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



TEST_F(CUDATest, GenerateGrids) {
  // SetUp
  const int size = 2;
  const float start = 1.0f;
  const float end = 2.0f;

  float *vectorX;
  float *vectorY;
  float *gridX;
  float *gridY;

  cudaMallocManaged(&vectorX, size * sizeof(float));
  cudaMallocManaged(&vectorY, size * sizeof(float));
  cudaMallocManaged(&gridX, size * size * sizeof(float));
  cudaMallocManaged(&gridY, size * size * sizeof(float));

  // Act
  linspaceCuda(vectorX, size, start, end);
  linspaceCuda(vectorY, size, start, end);
  createGridCuda(vectorX, vectorY, gridX, gridY, size);

  // Assert
  float expectedGridX[4] = {1.0f, 1.0f, 2.0f, 2.0f};
  float expectedGridY[4] = {1.0f, 2.0f, 1.0f, 2.0f};

  ASSERT_TRUE(std::equal(gridX, gridX + size * size, expectedGridX));
  ASSERT_TRUE(std::equal(gridY, gridY + size * size, expectedGridY));

  // TearDown
  cudaFree(vectorX);
  cudaFree(vectorY);
  cudaFree(gridX);
  cudaFree(gridY);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}