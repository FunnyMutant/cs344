#include "utils.h"
#include <stdio.h>

__global__ void reduceMin(float* d_min, const float* const d_in)
{
   extern __shared__ float shmem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   size_t shmemIdx = threadIdx.x;
   shmem[shmemIdx] = d_in[threadId];
   __syncthreads();
   
   int sold = blockDim.x;
   for (int step = blockDim.x >> 1; step > 0; step >>= 1)
   {
      if (shmemIdx < step)
      {
         shmem[shmemIdx] = min(shmem[shmemIdx], shmem[shmemIdx + step]);
		 if (shmemIdx == step - 1 && sold > 2 * step)
   		    shmem[shmemIdx] = min(shmem[shmemIdx], shmem[sold - 1]);
      }
	  
      __syncthreads();
	  sold = step;
   }

   if (0 == shmemIdx)
   {
      d_min[blockIdx.x] = shmem[0];
   }
}

__global__ void reduceMax(float* d_max, const float* const d_in)
{
   extern __shared__ float shmem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   size_t shmemIdx = threadIdx.x;
   shmem[shmemIdx] = d_in[threadId];
   __syncthreads();
   
   int sold = blockDim.x;
   for (int step = blockDim.x >> 1; step > 0; step >>= 1)
   {
      if (shmemIdx < step)
      {
         shmem[shmemIdx] = max(shmem[shmemIdx], shmem[shmemIdx + step]);
		 if (shmemIdx == step - 1 && sold > 2 * step)
   		    shmem[shmemIdx] = max(shmem[shmemIdx], shmem[sold - 1]);
      }
	  
      __syncthreads();
	  sold = step;
   }

   if (0 == shmemIdx)
   {
      d_max[blockIdx.x] = shmem[0];
   }
}

void reduceMinMax(float* d_min, float* d_max, const float* const d_in, size_t length)
{
   size_t threadsAmount = 1024;
   size_t blocks = length / threadsAmount;
   float* d_intermediate;
   cudaMalloc((void **) &d_intermediate, blocks * sizeof(float));
   
   reduceMin<<<blocks, threadsAmount, threadsAmount * sizeof(float)>>>(d_intermediate, d_in);
   checkCudaErrors(cudaGetLastError());
   reduceMin<<<1, blocks, blocks * sizeof(float)>>>(d_min, d_intermediate);
   checkCudaErrors(cudaGetLastError());
   cudaDeviceSynchronize();
   
   reduceMax<<<blocks, threadsAmount, threadsAmount * sizeof(float)>>>(d_intermediate, d_in);
   checkCudaErrors(cudaGetLastError());
   reduceMax<<<1, blocks, blocks * sizeof(float)>>>(d_max, d_intermediate);
   checkCudaErrors(cudaGetLastError());
   cudaFree(d_intermediate);
   cudaDeviceSynchronize();
   
}

__global__ void histoKernel(unsigned int* const d_out, const float* const d_in, float minLog, float maxLog)
{
   extern __shared__ unsigned int cdfMem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   
   cdfMem[threadIdx.x] = 0;
   __syncthreads();

   int bin = (d_in[threadId] - minLog) / (maxLog - minLog) * (float)blockDim.x;
   atomicAdd(&cdfMem[bin], 1);
   __syncthreads();
   atomicAdd(&d_out[threadIdx.x], cdfMem[threadIdx.x]);
}

__global__ void scanKernel(unsigned int* d_in)
{
   extern __shared__ unsigned int cdfMem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   cdfMem[threadId] = d_in[threadId];
   __syncthreads();
   
   
   int locId = threadIdx.x;
   for (int step = 1; step < blockDim.x; step <<= 1)
   {
      if (locId - step >= 0)
	  {
	     unsigned int curValue = cdfMem[locId - step];
         atomicAdd(&cdfMem[locId], curValue);
	  }
      __syncthreads();
   }
   if (0 == threadId)
   {
      d_in[0] = 0;
   }
   d_in[threadId] = cdfMem[threadId];
}

void histo(unsigned int* d_cdf, const float* const d_in, float min, float max, size_t lengthIn, size_t lengthOut)
{
   size_t threadsAmount = lengthOut;
   size_t blocks = lengthIn / lengthOut;
   histoKernel<<<blocks, threadsAmount, (threadsAmount + 1) * sizeof(unsigned int)>>>(d_cdf, d_in, min, max);
   checkCudaErrors(cudaGetLastError());
   cudaDeviceSynchronize();
   scanKernel<<<1, threadsAmount, threadsAmount * sizeof(unsigned int)>>>(d_cdf);
   checkCudaErrors(cudaGetLastError());
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
   
   histo(d_cdf, d_logLuminance, min_logLum, max_logLum, length, numBins);
   checkCudaErrors(cudaGetLastError());
   cudaDeviceSynchronize();
}
