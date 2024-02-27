#include <iostream>

#include <thrust/device_vector.h>

#ifndef UTILS_H
#define UTILS_H


void checkArrays(float *gridX, float *gridY, float *gridZ)
{
  /* Run a sanity check on the grids to ensure they hold during development. */

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

template <class T, class V>
void reshapeArray(T *flatArray, V **output, int m, int n)
{
  /*
  Convert a flat vector into a m x n matrix

  Arguments:
    flatArray: the array to be transformed.
    output: the matrix that will store the transformation.
    m: the number of rows.
    n: the number of columns
  */
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      output[i][j] = flatArray[i * m + j];
    }
  }
}

double** allocateMatrix(int rows, int cols)
{
  /*Allocate a matrix of the given size of rows & cols.*/
  double **output;
  cudaMallocManaged(&output, rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
    cudaMallocManaged(&output[i], cols * sizeof(double));
  }

  return output;
}

void freeMatrix(double** matrix, int rows)
{
  for (int i = 0; i < rows; i++) {
    cudaFree(matrix[i]);
  }
  cudaFree(matrix);
}

struct saxpy_functor
{
  const double a;  // class attribute

  saxpy_functor(double _a) : a(_a) {}  // Initialise

  // operator() is a special member of `struct` that allows this functor to be
  // used as a function. Here we are sort of overriding this member.
  __host__ __device__
      double operator()(const double& x, const double& y) const {
          return x * a;
      }
};

void saxpy_fast(
  double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y
  )
{
  /*
  Compute Y <- A * X + Y on the device using a functor.

  https://docs.nvidia.com/cuda/thrust/index.html#transformations
  https://chat.openai.com/c/ecd2c19e-4e91-44fc-9ff8-bbf1dca99eff
  */

  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

#endif