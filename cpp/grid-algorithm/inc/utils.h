#include <iostream>


#ifndef UTILS_H
#define UTILS_H

void printArray(float *arr, int n=3) {
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


void reshapeArray(float *flatArray, float **output, int m, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      output[i][j] = flatArray[i * m + j];
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << output[i][j] << ' ';
    }
    std::cout << '\n';
  }
}

#endif