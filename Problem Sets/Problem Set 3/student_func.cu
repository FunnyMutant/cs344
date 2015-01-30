#include "utils.h"
#include <stdio.h>

__global__ void reduceMin(float* d_min, const float* const d_in)
{
   extern __shared__ float shmem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   size_t shmemIdx = threadIdx.x;
   float curIn = shmem[shmemIdx] = d_in[threadId];
   __syncthreads();
   
   for (int step = blockDim.x >> 1; step > 0; step >>= 1)
   {
      if (shmemIdx < step)
      {
         shmem[shmemIdx] = min(shmem[shmemIdx + step], curIn);
      }
      __syncthreads();
   }
   if (0 == shmemIdx)
      d_min[blockIdx.x] = shmem[0];
}

__global__ void reduceMax(float* d_max, const float* const d_in)
{
   extern __shared__ float shmem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   size_t shmemIdx = threadIdx.x;
   float curIn = shmem[shmemIdx] = d_in[threadId];
   __syncthreads();
   for (int step = blockDim.x >> 1; step > 0; step >>= 1)
   {
      if (shmemIdx < step)
      {
         shmem[shmemIdx] = max(shmem[shmemIdx + step], curIn);
      }
      __syncthreads();
   }
   if (0 == shmemIdx)
      d_max[blockIdx.x] = shmem[0];
}

void reduceMinMax(float* d_min, float* d_max, const float* const d_in, size_t length)
{
   float* d_intermediate;
   cudaMalloc((void **) &d_intermediate, length * sizeof(float));
   size_t threadsAmount = length >> 10;
   size_t blocks = length / threadsAmount;
   reduceMin<<<blocks, threadsAmount, threadsAmount * sizeof(float)>>>(d_intermediate, d_in);
   reduceMin<<<1, blocks, blocks * sizeof(float)>>>(d_min, d_intermediate);
   
   reduceMax<<<blocks, threadsAmount, threadsAmount * sizeof(float)>>>(d_intermediate, d_in);
   reduceMax<<<1, blocks, blocks * sizeof(float)>>>(d_max, d_intermediate);
   
   cudaFree(d_intermediate);
}

__global__ void histoKernel(unsigned int* const d_out, const float* const d_in, float minLog, float maxLog)
{
   extern __shared__ float shmem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   size_t shmemIdx = threadIdx.x;
   float curIn = d_in[threadId];
   if(curIn < -3.109206)
      printf("min not found %f\n", curIn);
   if(curIn > -0.518340)
      printf("max not found %f\n", curIn);

   int result = ((curIn - minLog) / (maxLog - minLog)) * blockDim.x;
   
   atomicAdd(&shmem[min((int)blockDim.x - 1, result)], 1);
   __syncthreads();

   if (0 != shmemIdx)
      atomicAdd(&d_out[shmemIdx], shmem[shmemIdx]);
}


void histo(unsigned int *d_cdf, const float* const d_in, float min, float max, size_t lengthIn, size_t lengthOut)
{
   size_t threadsAmount = lengthOut;
   size_t blocks = lengthIn / lengthOut;
   histoKernel<<<blocks, threadsAmount, lengthOut * sizeof(float)>>>(d_cdf, d_in, min, max);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   float* d_min, *d_max;
   checkCudaErrors(cudaMalloc(&d_min,   sizeof(float)));
   checkCudaErrors(cudaMemcpy(d_min, &min_logLum, sizeof(float), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMalloc(&d_max,   sizeof(float)));
   checkCudaErrors(cudaMemcpy(d_max, &max_logLum, sizeof(float), cudaMemcpyHostToDevice));
   
   const size_t length = numRows * numCols;
   reduceMinMax(d_min, d_max, d_logLuminance, length);
   checkCudaErrors(cudaGetLastError());
   checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
   cudaFree(d_min);
   checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
   cudaFree(d_max);
   
   printf("min=%f\tmax=%f\n", min_logLum, max_logLum);
   
   histo(d_cdf, d_logLuminance, min_logLum, max_logLum, length, numBins);
   checkCudaErrors(cudaGetLastError());
}
