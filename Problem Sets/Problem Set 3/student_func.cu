#include "utils.h"
#include <stdio.h>

__global__ void calc(const float* const d_in, int *d_bins, unsigned int* const d_cdf, float* d_min, float* d_max, size_t numRows, size_t numCols, size_t numBins)
{
   /*
   size_t threadPos = threadIdx.x + blockDim.x * blockIdx.x;
   float curIn = d_in[threadPos];
   //step 1
   while(*d_min > curIn)
   {
      *d_min = curIn;
      __threadfence();
   }
   while(*d_max < curIn)
   {
      *d_max = curIn;
      __threadfence();
   }
   float curMin = *d_min, curMax = *d_max;
   
   printf("step 2");
   //step 2
   float logLumRange = curMax - curMin;

   //step 3
   printf("step 3");
   size_t bin = min((unsigned int)(numBins - 1), (unsigned int)((curIn - curMin) / logLumRange * numBins));
   atomicAdd(&d_bins[bin], 1);
   __syncthreads();
   
   //step 4
   printf("step 4");
   if (0 == threadPos || threadPos >= numBins)
      return;
   d_cdf[threadPos] = d_cdf[threadPos - 1] + d_bins[threadPos - 1];
   */
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   int* d_bins;
   float* d_min, *d_max;
   checkCudaErrors(cudaMalloc(&d_bins,   sizeof(int) * numBins));
   checkCudaErrors(cudaMemset(d_bins, 0,   sizeof(int) * numBins));
   checkCudaErrors(cudaMalloc(&d_min,   sizeof(float)));
   checkCudaErrors(cudaMemset(d_min, 0,   sizeof(float)));
   checkCudaErrors(cudaMalloc(&d_max,   sizeof(float)));
   checkCudaErrors(cudaMemset(d_max, 0,   sizeof(float)));
   
   const size_t length = numRows * numCols;
   printf("%i\t%i\t%i\n", numBins, length, length / numBins);
   calc<<<numBins, length / numBins>>>(d_logLuminance, d_bins, d_cdf, &min_logLum, &max_logLum, numRows, numCols, numBins);
   checkCudaErrors(cudaGetLastError());
   //min_logLum = *d_min;
   //max_logLum = *d_max;
   //checkCudaErrors(cudaMemcpy(d_min, &min_logLum, sizeof(float), cudaMemcpyDeviceToHost));
   //checkCudaErrors(cudaMemcpy(d_max, &max_logLum, sizeof(float), cudaMemcpyDeviceToHost));
   
   cudaFree(d_bins);
   cudaFree(d_min);
   cudaFree(d_max);
}
