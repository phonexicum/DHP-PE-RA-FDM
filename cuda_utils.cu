#include "cuda_utils.h"

// ==================================================================================================================================================
//                                                                                                                                               F fi
// ==================================================================================================================================================
__device__ double fi (const double x, const double y) {
    return logf(1 + x*y);
}

__device__ double F (const double x, const double y) {
    return (x*x + y*y)/((1 + x*y)*(1 + x*y));
}


// ==================================================================================================================================================
//                                                                                                                            cudakernel_MemsetDouble
// ==================================================================================================================================================
__global__ void cudakernel_MemsetDouble (double* const f, const double value, const double arr_size) {

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; k < arr_size; k += threadNum)
        f[k] = value;
}
