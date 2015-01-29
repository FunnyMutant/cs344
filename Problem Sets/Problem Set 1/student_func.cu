#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

#define COLOR_R_RATIO 0.299f
#define COLOR_G_RATIO 0.587f
#define COLOR_B_RATIO 0.114f

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int offset = numRows * blockIdx.y + blockIdx.x;

  uchar4 pixel = rgbaImage[offset];
  float partR = COLOR_R_RATIO * (float)pixel.x;
  float partG = COLOR_G_RATIO * (float)pixel.y;
  float partB = COLOR_B_RATIO * (float)pixel.z;
  
  greyImage[offset] = partR + partG + partB;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 blockSize(1, 1, 1);
  const dim3 gridSize(numRows, numCols, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
