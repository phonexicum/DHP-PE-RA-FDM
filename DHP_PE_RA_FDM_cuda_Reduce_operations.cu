#include <utility>

using std::pair;
using std::swap;

#include "DHP_PE_RA_FDM.h"


// ==================================================================================================================================================
//                                                                                                                    cudakernel_PrepareScalarProduct
// ==================================================================================================================================================
__global__ void cudakernel_PrepareScalarProduct (double* const cuda_sum_aggr_arr, const double* const f1, const double* const f2,
    const int arr_size, const int hx, const int hy){

    int threadId = THREAD_IN_GRID_ID;

    if (threadId < arr_size)
        cuda_sum_aggr_arr[threadId] = f1[threadId] * f2[threadId] * hx * hy;
}


// ==================================================================================================================================================
//                                                                                                                              cudakernel_ComputeSum
// ==================================================================================================================================================
// 
// This cuda kernel summarize `blockSize` values from function f corresponding for current block and stores the result into
//      `cuda_sum_aggr_arr[blockLinearId]`
// 
__global__ void cudakernel_ComputeSum (double* const cuda_sum_aggr_arr, const double* const f, const int arr_size){

    const int blockSize = BLOCK_SIZE;
    int thisBlockSize;
    
    extern __shared__ double data []; // shared memory in amount of 1024 doubles (1024=maxThreadsPerBlock)

    int threadLinearBlockId = THREAD_IN_BLOCK_ID;
    int blockLinearId = BLOCK_IN_GRID_ID;

    // Cut off redundant blocks
    if (blockLinearId < (arr_size -1) / blockSize +1){

        bool lastBlock = blockLinearId == blockSize -1;
        if (lastBlock)
            thisBlockSize = arr_size % blockSize;
        else
            thisBlockSize = blockSize;

        if (not lastBlock or (lastBlock and (threadLinearBlockId < thisBlockSize)) ){

            data[threadLinearBlockId] = f[threadLinearBlockId + blockLinearId * blockSize];
            __syncthreads ();

            for (int s = 1; s < thisBlockSize; s *= 2){
                if (threadLinearBlockId % (2*s) == 0 and threadLinearBlockId + s < thisBlockSize){
                    data[threadLinearBlockId] += data[threadLinearBlockId + s];
                }
                __syncthreads ();
            }

            if (threadLinearBlockId == 0)
                cuda_sum_aggr_arr[blockLinearId] = data[0];
        }
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


    int dimension = procCoords.x_cells_num * procCoords.y_cells_num;
    pair<dim3, dim3> mesh = GridDistribute(dimension);
    cudakernel_PrepareScalarProduct<<<mesh.first, mesh.second>>> (cuda_sum_aggr_arr1, f1, f2, dimension, hx, hy);

    while (true){
        cudakernel_ComputeSum<<<mesh.first, mesh.second, devProp.maxThreadsPerBlock * sizeof(*cuda_sum_aggr_arr1)>>> (cuda_sum_aggr_arr2, cuda_sum_aggr_arr1, dimension);

        if (dimension == 1) break;
        
        dimension = mesh.second.x * mesh.second.y * mesh.second.z; // number of blocks
        mesh = GridDistribute(dimension);
        swap(cuda_sum_aggr_arr1, cuda_sum_aggr_arr2);
    }

    double scalar_product = 0;
    SAFE_CUDA(cudaMemcpy(&scalar_product, cuda_sum_aggr_arr2, sizeof(double), cudaMemcpyDeviceToHost));

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
    const int arr_size){

    int threadId = THREAD_IN_GRID_ID;

    if (threadId < arr_size)
        cuda_sum_aggr_arr[threadId] = fabs(f1[threadId] - f2[threadId]);
}


// ==================================================================================================================================================
//                                                                                                                              cudakernel_ComputeMax
// ==================================================================================================================================================
// 
// This cuda kernel found maximum in `blockSize` values from function f corresponding for current block and stores the result into
//      `cuda_sum_aggr_arr[blockLinearId]`
// 
__global__ void cudakernel_ComputeMax (double* const cuda_sum_aggr_arr, const double* const f, const int arr_size){

    const int blockSize = BLOCK_SIZE;
    int thisBlockSize;
    
    extern __shared__ double data []; // shared memory in amount of 1024 doubles (1024=maxThreadsPerBlock)

    int threadLinearBlockId = THREAD_IN_BLOCK_ID;
    int blockLinearId = BLOCK_IN_GRID_ID;

    // Cut off redundant blocks
    if (blockLinearId < (arr_size -1) / blockSize +1) {

        bool lastBlock = blockLinearId == blockSize -1;
        if (lastBlock)
            thisBlockSize = arr_size % blockSize;
        else
            thisBlockSize = blockSize;

        if (not lastBlock or (lastBlock and (threadLinearBlockId < thisBlockSize)) ){

            data[threadLinearBlockId] = f[threadLinearBlockId + blockLinearId * blockSize];
            __syncthreads ();

            for (int s = 1; s < thisBlockSize; s *= 2){
                if (threadLinearBlockId % (2*s) == 0 and threadLinearBlockId + s < thisBlockSize){
                    data[threadLinearBlockId] = max(data[threadLinearBlockId], data[threadLinearBlockId + s]);
                }
                __syncthreads ();
            }

            if (threadLinearBlockId == 0)
                cuda_sum_aggr_arr[blockLinearId] = data[0];
        }
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


    int dimension = procCoords.x_cells_num * procCoords.y_cells_num;
    pair<dim3, dim3> mesh = GridDistribute(dimension);
    cudakernel_PrepareStopCriteria<<<mesh.first, mesh.second>>> (cuda_sum_aggr_arr1, f1, f2, dimension);

    while (true){
        cudakernel_ComputeMax<<<mesh.first, mesh.second, devProp.maxThreadsPerBlock * sizeof(*cuda_sum_aggr_arr1)>>> (cuda_sum_aggr_arr2, cuda_sum_aggr_arr1, dimension);

        if (dimension == 1) break;
        
        dimension = mesh.second.x * mesh.second.y * mesh.second.z; // number of blocks
        mesh = GridDistribute(dimension);
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
