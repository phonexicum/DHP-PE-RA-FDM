#pragma once

#include <algorithm>
#include <utility>

using std::min;
using std::pair;
using std::make_pair;
using std::swap;

#include <cuda_runtime.h>


#include <iostream>
using std::cout;
using std::endl;


// general case
// #define THREAD_IN_GRID_ID  threadIdx.x + blockIdx.x * blockDim.x + blockDim.x * gridDim.x * (threadIdx.y + blockIdx.y * blockDim.y + blockDim.y * gridDim.y * (threadIdx.z + blockIdx.z * blockDim.z))
// #define THREAD_IN_BLOCK_ID threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
// #define BLOCK_IN_GRID_ID   blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)

// special case (my case) (for my `GridDistribute` function)
#define THREAD_IN_GRID_ID  threadIdx.x + blockIdx.x * blockDim.x
#define THREAD_IN_BLOCK_ID threadIdx.x
#define BLOCK_IN_GRID_ID   blockIdx.x


#define BLOCK_SIZE blockDim.x * blockDim.y * blockDim.z

__device__ double fi (const double x, const double y);

__device__ double F (const double x, const double y);

// #define fi(x, y) logf(1 + x*y)
// #define F(x, y) (x*x + y*y)/((1 + x*y)*(1 + x*y))

__global__ void cudakernel_MemsetDouble (double* const f, const double value, const double arr_size);
