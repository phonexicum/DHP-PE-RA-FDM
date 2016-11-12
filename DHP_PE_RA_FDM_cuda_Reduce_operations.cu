#include <utility>

using std::pair;

#include "DHP_PE_RA_FDM.h"


// ==================================================================================================================================================
//                                                                                                                               cudakernel_Compute_r
// ==================================================================================================================================================
__global__ void cudakernel_Compute_r (double* const r, const double* const delta_p, const ProcComputingCoords procCoords,
    const int X1, const int Y1, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = threadLinearIdx % (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
    int j = threadLinearIdx / (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

    r[j * procCoords.x_cells_num + i] =
        delta_p[j * procCoords.x_cells_num + i] -
        F(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy)
    ;
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_r
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Compute_r (double* const r, const double* const delta_p) const{

    // internal region
    pair<dim3, dim3> mesh = GridDistribute(
        (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top)) *
        (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left))
    );
    cudakernel_Compute_r<<<mesh.first, mesh.second, 0, cudaStreams[0]>>> (r, delta_p, procCoords, X1, Y1, hx, hy);

    cudaAllStreamsSynchronize(0, 0);
}


// ==================================================================================================================================================
//                                                                                                                    cudakernel_ComputeScalarProduct
// ==================================================================================================================================================
__global__ void cudakernel_ComputeScalarProduct (const double* const f, double* const scalar_product_aggregation_array, const int step){

    __shared__ int data [blockDim.x * blockDim.y * blockDim.z];

    // int threadLinearGridIdx = ((threadIdx.z + blockIdx.z * blockDim.z) * blockDim.y * gridDim.y + threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x + threadIdx.x + blockIdx.x * blockDim.x;

    int threadLinearBlockIdx = threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y);
    int blockLinearIdx = blockIdx.x + gridDim.x * (blockIdx.y + blockIdx.z * gridDim.y);
    // int threadLinearGridIdx = threadLinearBlockIdx + blockLinearIdx * blockDim.x * blockDim.y * blockDim.z;


    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x data [tid] = inData [i];

    __syncthreads ();
    for (int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2*s) == 0){
            data [tid] += data [tid + s];
        }
        __syncthreads ();
    }

    if (tid == 0){
        outData [blockIdx.x] = data [0];
    }
}




// ==================================================================================================================================================
//                                                                                                         DHP_PE_RA_FDM::cuda_ComputingScalarProduct
// ==================================================================================================================================================
double DHP_PE_RA_FDM::cuda_ComputingScalarProduct(const double* const f1, const double* const f2){

    if (scalar_product_aggregation_array == NULL)
        SAFE_CUDA(cudaMalloc(&scalar_product_aggregation_array,
            procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*scalar_product_aggregation_array)));


    double scalar_product = 0;
    for (int j = 0; j < procCoords.y_cells_num; j++){
        for (int i = 0; i < procCoords.x_cells_num; i++){
            scalar_product += hx * hy * f1[j * procCoords.x_cells_num + i] * f2[j * procCoords.x_cells_num + i];
        }
    }

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
//                                                                                                                   DHP_PE_RA_FDM::cuda_StopCriteria
// ==================================================================================================================================================
bool DHP_PE_RA_FDM::cuda_StopCriteria(const double* const f1, const double* const f2){

    double norm = 0;
    for (int i = 0; i < procCoords.x_cells_num * procCoords.y_cells_num; i++){
        norm = max(norm, abs(f1[i] - f2[i]));
    }

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