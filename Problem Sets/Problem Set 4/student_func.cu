#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

typedef unsigned int uint32;

const size_t BITS_PER_BYTE = 8;
const size_t BYTES_AMOUNT = sizeof(unsigned int);
const size_t BITS_AMOUNT = BITS_PER_BYTE * BYTES_AMOUNT;
const size_t THREADS_AMOUNT = 320;

uint32*				h_goldMem;
uint32*				h_inVals;
uint32*				h_tempMem;
uint32* 			d_tempMem;

__global__ void prescan(uint32* g_odata, uint32* g_idata, int bit){

    extern __shared__ uint32 temp[];// allocated on invocation
    const uint32 thid = threadIdx.x;
	const uint32 doubleId = thid << 1;
	const uint32 n = blockDim.x * gridDim.x;
    uint32 offset = 1;
    temp[doubleId] = (g_idata[doubleId] >> bit) & 1; // load input into shared memory
    temp[doubleId + 1] = (g_idata[doubleId + 1] >> bit) & 1;


    for (int d = n >> 1; d > 0; d >>= 1)
	{ // build sum in place up the tree
        __syncthreads();
        if (thid < d)
		{
            int ai = offset * (doubleId + 1) - 1;
            int bi = offset * (doubleId + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thid == 0)
	{
		temp[n - 1] = 0;
	} // clear the last element

    for (int d = 1; d < n; d *= 2)
	{ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d)
		{
            int ai = offset * (doubleId + 1) - 1;
            int bi = offset * (doubleId + 2) - 1;
            uint32 t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    g_odata[doubleId] = temp[doubleId]; // write results to device memory
    g_odata[doubleId + 1] = temp[doubleId + 1];
}

__global__ void partition_by_bit(uint32* const inVals, uint32* const inPos, uint32* scanResult, uint32 bit)
{
    uint32 threadId = threadIdx.x + blockDim.x * blockIdx.x;
    uint32 size = blockDim.x * gridDim.x;
    uint32 x_i = inVals[threadId];
	uint32 pos_i = inPos[threadId];
    uint32 p_i = (x_i >> bit) & 1u;

    __syncthreads();

    uint32 T_before = scanResult[threadId];
	__syncthreads();
    uint32 F_total  = size - scanResult[size - 1];
    __syncthreads();
	
	uint32 newPos = 0;
	if (p_i > 0)
		newPos = T_before - 1 + F_total;
	else
		newPos = threadId - T_before;
	
	if (newPos >= size)
		printf("newPos is invalid: newPos=%u\n", newPos);
	
	inVals[newPos] = x_i;
	inPos[newPos] = pos_i;
}

void your_sort(uint32* const d_inputVals,
               uint32* const d_inputPos,
               uint32* const d_outputVals,
               uint32* const d_outputPos,
               const size_t numElems)
{
	uint32 *d_tempMem;
	
	size_t elemsAmount = numElems;
	printf("Elements amount is %i\n", elemsAmount);
	
	h_inVals = (uint32*)malloc(elemsAmount * BYTES_AMOUNT);
	checkCudaErrors(cudaMemcpy(h_inVals, d_inputVals, elemsAmount * BYTES_AMOUNT, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaMalloc(&d_tempMem, elemsAmount * BYTES_AMOUNT));
	
	h_tempMem = (uint32*)malloc(elemsAmount * BYTES_AMOUNT);
	h_goldMem = (uint32*)malloc(elemsAmount * BYTES_AMOUNT);
	memset(h_goldMem, 0, elemsAmount * BYTES_AMOUNT);
	
	
	const size_t blocksAmount = elemsAmount / THREADS_AMOUNT;
    for(uint32 bit = 0; bit < BITS_AMOUNT; bit++ )
    {
		checkCudaErrors(cudaMemset(d_tempMem, 0, elemsAmount * BYTES_AMOUNT));
		printf("processing bit %i\n", bit);
		
		prescan<<<blocksAmount, THREADS_AMOUNT, blocksAmount * BITS_AMOUNT * BYTES_AMOUNT * 2>>>(d_tempMem, d_inputVals, bit);
		//plus_scan<<<blocksAmount, THREADS_AMOUNT, THREADS_AMOUNT * BYTES_AMOUNT>>>(d_tempMem, d_inputVals, bit);
		cudaDeviceSynchronize();
		
		cudaMemcpy(h_tempMem, d_tempMem, elemsAmount * BYTES_AMOUNT, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		for (uint32 i = 1; i < elemsAmount; ++i)
			h_goldMem[i] = h_goldMem[i - 1] + ((h_inVals[i - 1] >> bit) & 1);
/*
		printf("\n");
		for(uint32 i = 0; i < elemsAmount; ++i)
			printf("in[%i]=%i\t", i, ((h_inVals[i] >> bit) & 1));
		printf("\n");
		for(uint32 i = 0; i < elemsAmount; ++i)
			printf("out[%i]=%i\t", i, h_tempMem[i]);
		printf("\n");
		for(uint32 i = 0; i < elemsAmount; ++i)
			printf("gold[%i]=%i\t", i, h_goldMem[i]);
		printf("\n");
		
		for(uint32 i = 0; i < elemsAmount; ++i)
			if (h_goldMem[i] != h_tempMem[i])
				printf("invalid value in %i\n", i);
		printf("\n");
		printf("\n");
*/		
		partition_by_bit<<<blocksAmount, THREADS_AMOUNT>>>(d_inputVals, d_inputPos, d_tempMem, bit);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
    }
	cudaMemcpy(d_outputVals, d_inputVals, elemsAmount * BYTES_AMOUNT, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_outputPos, d_inputPos, elemsAmount * BYTES_AMOUNT, cudaMemcpyDeviceToDevice);
	
	free(h_inVals);
	free(h_goldMem);
	free(h_tempMem);
	cudaFree(d_tempMem);
}
