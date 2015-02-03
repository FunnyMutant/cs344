#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

typedef unsigned int uint32;

const size_t BITS_PER_BYTE = 8;
const size_t BYTES_AMOUNT = sizeof(unsigned int);
const size_t BITS_AMOUNT = BITS_PER_BYTE * BYTES_AMOUNT;
const size_t THREADS_AMOUNT = 320;

__device__ void plusScan(uint32* const d_values)
{
   uint32 elemsAmount = gridDim.x * blockDim.x;
   uint32 threadId = threadIdx.x + blockDim.x * blockIdx.x;
   
   __syncthreads();

   for (size_t step = 1; step < elemsAmount; step <<= 1)
   {
      uint32 temp;
      if (threadId >= step)
         temp = d_values[threadId - step];
         
      __syncthreads();
      
      if (threadId >= step)
         d_values[threadId] += temp;

      __syncthreads();
   }
}

__global__ void radixSort( uint32* const d_inputVals,
                           uint32* const d_inputPos,
                           uint32* const d_outputVals,
                           uint32* const d_outputPos)
{
   size_t elemsAmount = gridDim.x * blockDim.x;
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;

   for (size_t bit = 0; bit < BITS_AMOUNT; bit++)
   {
      uint32 curValue = d_inputVals[threadId];
      uint32 curPos = d_inputPos[threadId];
      uint32 curBit = (curValue >> bit) & 1;
      d_inputVals[threadId] = curBit;

      plusScan(d_inputVals);
      
      __syncthreads();

      uint32 True_before = d_inputVals[threadId];
      uint32 True_total  = d_inputVals[elemsAmount - 1];
      uint32 False_total  = elemsAmount - True_total;
      __syncthreads();
      
      int newPos = curBit ? ((True_before + False_total) - 1) : threadId - True_before;
      d_inputVals[newPos] = curValue;
      d_inputPos[newPos] = curPos;
   }
   d_outputVals[threadId] = d_inputVals[threadId];
   d_outputPos[threadId] = d_inputPos[threadId];
}

void your_sort(uint32* const d_inputVals,
               uint32* const d_inputPos,
               uint32* const d_outputVals,
               uint32* const d_outputPos,
               const size_t numElems)
{
   cudaMemset(d_outputVals, 0, BYTES_AMOUNT * numElems);
   cudaMemset(d_outputPos, 0, BYTES_AMOUNT * numElems);
   const size_t blocksAmount = numElems / THREADS_AMOUNT;
   radixSort<<<blocksAmount, THREADS_AMOUNT>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos);
   checkCudaErrors(cudaGetLastError());
   
   for (size_t i = 0; i < blocksAmount; i++)
   {
      for (size_t j = 0; j < THREADS_AMOUNT; j++)
      {
         printf("%i\t", d_inputVals[i * blocksAmount + j]);
      }
      printf("\n");
   }
   
   printf("mission accomplished\n");
}
