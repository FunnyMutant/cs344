#include "reference_calc.cpp"
#include "utils.h"

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT BLOCK_WIDTH

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  __shared__ float filterCache[BLOCK_WIDTH * BLOCK_HEIGHT];
  //__shared__ float channelCache[BLOCK_WIDTH * BLOCK_HEIGHT];
  const int threadPos = blockDim.x * threadIdx.y + threadIdx.x;
  filterCache[threadPos] = filter[threadPos];
  __syncthreads();

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  float result = 0.f;
  const int filterHalf = filterWidth / 2;
  for (int filter_r = -filterHalf; filter_r <= filterHalf; ++filter_r) {
	for (int filter_c = -filterHalf; filter_c <= filterHalf; ++filter_c) {
	  int image_r = min(max(thread_2D_pos.y + filter_r, 0), numRows - 1);
	  int image_c = min(max(thread_2D_pos.x + filter_c, 0), numCols - 1);

	  float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
	  float filter_value = filterCache[(filter_r + filterHalf) * filterWidth + (filter_c + filterHalf)];
	  result += image_value * filter_value;
	}
  }
  outputChannel[thread_1D_pos] = result;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  const uchar4 rgba = inputImageRGBA[thread_1D_pos];
  redChannel[thread_1D_pos] = rgba.x;
  greenChannel[thread_1D_pos] = rgba.y;
  blueChannel[thread_1D_pos] = rgba.z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  unsigned int bytesAmount = sizeof(float) * filterWidth * filterWidth;
  checkCudaErrors(cudaMalloc(&d_filter, bytesAmount));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, bytesAmount, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 globalGridSize(numCols, numRows, 1);
  separateChannels<<<globalGridSize, 1>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  const dim3 localGridSize(numCols, numRows, 1);
  gaussian_blur<<<localGridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<localGridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<localGridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  recombineChannels<<<globalGridSize, 1>>>(d_redBlurred,
                                           d_greenBlurred,
                                           d_blueBlurred,
                                           d_outputImageRGBA,
                                           numRows,
                                           numCols);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
