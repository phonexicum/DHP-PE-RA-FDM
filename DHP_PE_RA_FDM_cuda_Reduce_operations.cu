#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                                    cudakernel_PrepareScalarProduct
// ==================================================================================================================================================
__global__ void cudakernel_PrepareScalarProduct (double* const cuda_sum_aggr_arr, const double* const f1, const double* const f2,
    const int arr_size, const double hxhy, const ProcComputingCoords procCoords){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; true; k += threadNum) {

        int i = static_cast<int>(procCoords.left) + k % (
            procCoords.x_cells_num - static_cast<int>(procCoords.left) - static_cast<int>(procCoords.right));
        
        int j = static_cast<int>(procCoords.top) + k / (
            procCoords.x_cells_num - static_cast<int>(procCoords.left) - static_cast<int>(procCoords.right));

        if (j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom)) {
            cuda_sum_aggr_arr[k] = f1[j * procCoords.x_cells_num + i] * f2[j * procCoords.x_cells_num + i] * hxhy;
        } else {
            break;
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                              cudakernel_ComputeSum
// ==================================================================================================================================================
// 
// This cuda kernel summarize `blockSize` values from function f corresponding for current block and stores the result into
//      `cuda_sum_aggr_arr[blockLinearId]`
// 
__global__ void cudakernel_ComputeSum (double* const cuda_sum_aggr_arr, const double* const f, const int arr_size, const int demandedBlocksNumber){

    extern __shared__ double data []; // shared memory in amount of 1024 doubles (1024=maxThreadsPerBlock)

    // Cut off redundant blocks
    int blockLinearId = BLOCK_IN_GRID_ID;
    if (blockLinearId < demandedBlocksNumber) {

        int threadId = THREAD_IN_GRID_ID;
        int threadNum = GRID_SIZE_IN_THREADS;
        double sum = 0;
        for (int k = threadId; k < arr_size; k += threadNum)
            sum += f[k];

        int threadLinearBlockId = THREAD_IN_BLOCK_ID;
        data[threadLinearBlockId] = sum;
        __syncthreads ();

        int blockSize = BLOCK_SIZE;
        for (int s = 1; s < blockSize; s *= 2){
            if (threadLinearBlockId % (2*s) == 0 and threadLinearBlockId + s < blockSize){
                data[threadLinearBlockId] += data[threadLinearBlockId + s];
            }
            __syncthreads ();
        }

        if (threadLinearBlockId == 0)
            cuda_sum_aggr_arr[blockLinearId] = data[0];
    }
}


// ==================================================================================================================================================
//                                                                                                         DHP_PE_RA_FDM::cuda_ComputingScalarProduct
// ==================================================================================================================================================
double DHP_PE_RA_FDM::cuda_ComputingScalarProduct(const double* const f1, const double* const f2){

    if (cuda_sum_aggr_arr1 == NULL)
        SAFE_CUDA(cudaMalloc(&cuda_sum_aggr_arr1, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*cuda_sum_aggr_arr1)));
    if (cuda_sum_aggr_arr2 == NULL)
        SAFE_CUDA(cudaMalloc(&cuda_sum_aggr_arr2, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*cuda_sum_aggr_arr2)));
    
    int dimension = (procCoords.x_cells_num - static_cast<int>(procCoords.left) - static_cast<int>(procCoords.right)) *
        (procCoords.y_cells_num - static_cast<int>(procCoords.top) - static_cast<int>(procCoords.bottom));
    GridDistribute mesh (devProp, dimension);
    cudakernel_PrepareScalarProduct<<<mesh.gridDim, mesh.blockDim>>> (
        cuda_sum_aggr_arr1, f1, f2, dimension, hxhy, procCoords); CUDA_CHECK_LAST_ERROR;

    while (true) {

        cudakernel_ComputeSum<<<mesh.gridDim, mesh.blockDim, mesh.demandedThreadsPerBlock * sizeof(*cuda_sum_aggr_arr1)>>> (
            cuda_sum_aggr_arr2, cuda_sum_aggr_arr1, dimension, mesh.demandedBlocksNumber); CUDA_CHECK_LAST_ERROR;
        
        dimension = mesh.demandedBlocksNumber; // number of blocks
        if (dimension == 1)
            break;

        mesh = GridDistribute (devProp, dimension);
        swap(cuda_sum_aggr_arr1, cuda_sum_aggr_arr2);
    }

    double scalar_product = 0;
    SAFE_CUDA(cudaMemcpy(&scalar_product, cuda_sum_aggr_arr2, sizeof(scalar_product), cudaMemcpyDeviceToHost));

    double global_scalar_product = 0;

    int ret = MPI_Allreduce(
        &scalar_product,            // const void *sendbuf,
        &global_scalar_product,     // void *recvbuf,
        1,                          // int count,
        MPI_DOUBLE,                 // MPI_Datatype datatype,
        MPI_SUM,                    // MPI_Op op,
        procParams.comm             // MPI_Comm comm
    );
    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error reducing scalar_product.");

    return global_scalar_product;
}


// ==================================================================================================================================================
//                                                                                                                     cudakernel_PrepareStopCriteria
// ==================================================================================================================================================
__global__ void cudakernel_PrepareStopCriteria (double* const cuda_sum_aggr_arr, const double* const f1, const double* const f2,
    const int arr_size, const ProcComputingCoords procCoords){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; threadId < arr_size; k += threadNum) {

        int i = static_cast<int>(procCoords.left) + k % (
            procCoords.x_cells_num - static_cast<int>(procCoords.left) - static_cast<int>(procCoords.right));
        
        int j = static_cast<int>(procCoords.top) + k / (
            procCoords.x_cells_num - static_cast<int>(procCoords.left) - static_cast<int>(procCoords.right));

        if (j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom)) {
            cuda_sum_aggr_arr[k] = fabs(f1[j * procCoords.x_cells_num + i] - f2[j * procCoords.x_cells_num + i]);
        } else {
            break;
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                              cudakernel_ComputeMax
// ==================================================================================================================================================
// 
// This cuda kernel found maximum in `blockSize` values from function f corresponding for current block and stores the result into
//      `cuda_sum_aggr_arr[blockLinearId]`
// 
__global__ void cudakernel_ComputeMax (double* const cuda_sum_aggr_arr, const double* const f, const int arr_size, const int demandedBlocksNumber){

    extern __shared__ double data []; // shared memory in amount of 1024 doubles (1024=maxThreadsPerBlock)

    // Cut off redundant blocks
    int blockLinearId = BLOCK_IN_GRID_ID;
    if (blockLinearId < demandedBlocksNumber) {

        int threadId = THREAD_IN_GRID_ID;
        int threadNum = GRID_SIZE_IN_THREADS;
        double maximum = 0;
        for (int k = threadId; k < arr_size; k += threadNum)
            maximum = max(maximum, f[k]);

        int threadLinearBlockId = THREAD_IN_BLOCK_ID;
        data[threadLinearBlockId] = maximum;
        __syncthreads ();

        int blockSize = BLOCK_SIZE;
        for (int s = 1; s < blockSize; s *= 2){
            if (threadLinearBlockId % (2*s) == 0 and threadLinearBlockId + s < blockSize){
                data[threadLinearBlockId] = max(data[threadLinearBlockId], data[threadLinearBlockId + s]);
            }
            __syncthreads ();
        }

        if (threadLinearBlockId == 0)
            cuda_sum_aggr_arr[blockLinearId] = data[0];
    }
}


// ==================================================================================================================================================
//                                                                                                                   DHP_PE_RA_FDM::cuda_StopCriteria
// ==================================================================================================================================================
bool DHP_PE_RA_FDM::cuda_StopCriteria(const double* const f1, const double* const f2){

    if (cuda_sum_aggr_arr1 == NULL)
        SAFE_CUDA(cudaMalloc(&cuda_sum_aggr_arr1, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*cuda_sum_aggr_arr1)));
    if (cuda_sum_aggr_arr2 == NULL)
        SAFE_CUDA(cudaMalloc(&cuda_sum_aggr_arr2, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*cuda_sum_aggr_arr2)));


    int dimension = (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left)) *
        (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top));
    GridDistribute mesh (devProp, dimension);
    cudakernel_PrepareStopCriteria<<<mesh.gridDim, mesh.blockDim>>> (cuda_sum_aggr_arr1, f1, f2, dimension, procCoords); CUDA_CHECK_LAST_ERROR;

    while (true) {

        cudakernel_ComputeMax<<<mesh.gridDim, mesh.blockDim, mesh.demandedThreadsPerBlock * sizeof(*cuda_sum_aggr_arr1)>>> (
            cuda_sum_aggr_arr2, cuda_sum_aggr_arr1, dimension, mesh.demandedBlocksNumber); CUDA_CHECK_LAST_ERROR;

        dimension = mesh.demandedBlocksNumber; // number of blocks
        if (dimension == 1)
            break;

        mesh = GridDistribute (devProp, dimension);
        swap(cuda_sum_aggr_arr1, cuda_sum_aggr_arr2);
    }

    double norm = 0;
    SAFE_CUDA(cudaMemcpy(&norm, cuda_sum_aggr_arr2, sizeof(double), cudaMemcpyDeviceToHost));

    double global_norm = 0;

    int ret = MPI_Allreduce(
        &norm,                      // const void *sendbuf,
        &global_norm,               // void *recvbuf,
        1,                          // int count,
        MPI_DOUBLE,                 // MPI_Datatype datatype,
        MPI_MAX,                    // MPI_Op op,
        procParams.comm             // MPI_Comm comm
    );
    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error reducing scalar_product.");

    return global_norm < eps;
}
