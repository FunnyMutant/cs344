#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
   extern __shared__ unsigned int shmem[];
   const size_t localId = threadIdx.x * numVals;
   for(size_t i = 0; i < numVals; ++i)
      shmem[localId + i] = 0;

   __syncthreads();

   size_t threadId = (threadIdx.x + blockDim.x * blockIdx.x) * numVals;
   for (size_t i = 0; i < numVals; ++i)
      atomicAdd(&shmem[vals[threadId + i]], 1);

   __syncthreads();

   for(size_t i = 0; i < numVals; ++i)
      atomicAdd(&histo[localId + i], shmem[localId + i]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
   const int threads = 256;
   const int values = numBins / threads;
   const int blocks = numElems / numBins;
   yourHisto<<<blocks, threads, (numBins) * sizeof(unsigned int)>>>(d_vals, d_histo, values);

   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
}
