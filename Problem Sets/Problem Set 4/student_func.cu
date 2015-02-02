#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

/* 
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const size_t VALUES_AMOUNT = 2;
const size_t BITS_AMOUNT = sizeof(unsigned int) << 3;
const size_t BINS_AMOUNT = BITS_AMOUNT * VALUES_AMOUNT;

__device__ unsigned int histoResult[BINS_AMOUNT];
__device__ unsigned int *scanResult = histoResult;

__global__ void scanKernel()
{
   for (int step = 1; step < BINS_AMOUNT; step <<= 1)
   {
      if ((int)threadIdx.x - step >= 0)
      {
         scanResult[threadIdx.x] += scanResult[threadIdx.x - step];
      }
      __syncthreads();
   }
   if (0 == threadIdx.x)
   {
      scanResult[0] = 0;
   }
}

__global__ void histoKernel(unsigned int* const d_in)
{
   extern __shared__ unsigned int shMem[];
   size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
   
   unsigned int curValue = d_in[threadId];
   for (size_t index = 0; index < BITS_AMOUNT; index++)
   {
      unsigned int offset = (curValue & 1u) + 1;
      unsigned int bin = index * offset;
      atomicAdd(&shMem[bin], 1);
      curValue >>= 1;
   }
   __syncthreads();
   
   if (threadIdx.x < BINS_AMOUNT)
   {
      atomicAdd(&histoResult[threadIdx.x], shMem[threadIdx.x]);
   }
}

const size_t NUM_OF_THREADS_HISTO = 320;

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
   histoKernel<<<numElems / NUM_OF_THREADS_HISTO, NUM_OF_THREADS_HISTO, BINS_AMOUNT * sizeof(unsigned int)>>>(d_inputVals);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   scanKernel<<<1, BINS_AMOUNT>>>();
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}
