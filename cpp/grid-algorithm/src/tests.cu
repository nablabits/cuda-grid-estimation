#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../inc/normal_kernel.h"

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}