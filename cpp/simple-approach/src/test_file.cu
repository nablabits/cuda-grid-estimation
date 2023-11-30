#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "../inc/add.h"


/*
This is the chatGPT solution, in principle it should be triggered with
nvcc -o test_executable test_file.cu -lgtest -lgtest_main -pthread
but `-pthread` is not supported by nvcc. If one removes that bit, one hits
a fatal error gtest/gtest.h: No such file or directory. I installed gtestlib-dev
that seemed to solve the issue.

https://chat.openai.com/c/2bd8791e-753c-441b-9673-61a860643ddd

How to run this test:
make all
./bin/test_file
*/

// Test fixture class
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
TEST_F(CUDATest, AddFunctionTest) {
    const int arraySize = 5;
    float *x, *y;
    cudaMallocManaged(&x, arraySize*sizeof(float));
    cudaMallocManaged(&y, arraySize*sizeof(float));

    // Initialize input arrays (you can use random values, etc.)
    for (int i = 0; i < arraySize; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Call the CUDA function
    add<<<1, 1>>>(arraySize, x, y);

    // Copy the result back to the host
    cudaDeviceSynchronize();

    // Perform assertions to check if the CUDA function worked as expected
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_EQ(y[i], 3.0f);
    }

    // Free memory
    cudaFree(x);
    cudaFree(y);
}

// Add more test cases as needed

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}