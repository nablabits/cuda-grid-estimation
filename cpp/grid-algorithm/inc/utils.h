#include <iostream>

#include <thrust/device_vector.h>

#ifndef UTILS_H
#define UTILS_H

void printArray(float *arr, int n=3) {
  /* Handy function to print arrays. */
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}

// TODO: I need a template for this
void printArrayd(double *arr, int n=3) {
  /* Handy function to print arrays. */
  for (int i = 0; i < n; i++) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
}


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


void reshapeArray(float *flatArray, double **output, int m, int n) {
  // TODO: this needs a docstring
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      output[i][j] = flatArray[i * m + j];
    }
  }

  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < m; ++j) {
  //     std::cout << output[i][j] << ' ';
  //   }
  //   std::cout << '\n';
  // }
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
  // https://docs.nvidia.com/cuda/thrust/index.html#transformations
  // https://chat.openai.com/c/ecd2c19e-4e91-44fc-9ff8-bbf1dca99eff
  // Y <- A * X + Y.
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

#endif