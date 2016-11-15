#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                                               cudakernel_Compute_r
// ==================================================================================================================================================
__global__ void cudakernel_Compute_r (double* const r, const double* const delta_p, const int arr_size, const ProcComputingCoords procCoords,
    const double X1, const double Y1, const double hx, const double hy){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; true; k += threadNum) {

        int i = static_cast<int>(procCoords.left) + k % (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

        int j = static_cast<int>(procCoords.top) + k / (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

        if (j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom)) {
            r[j * procCoords.x_cells_num + i] =
                delta_p[j * procCoords.x_cells_num + i] -
                F(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy)
            ;
        } else {
            break;
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::cuda_Compute_r
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Compute_r (double* const r, const double* const delta_p) const{

    // internal region
    int dimension = (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left)) *
        (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top));
    GridDistribute mesh (devProp, dimension);
    cudakernel_Compute_r<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>> (r, delta_p, dimension, procCoords, X1, Y1, hx, hy); CUDA_CHECK_LAST_ERROR;

    cudaAllStreamsSynchronize(0, 0);
}


// ==================================================================================================================================================
//                                                                                                                               cudakernel_Compute_g
// ==================================================================================================================================================
__global__ void cudakernel_Compute_g (double* const g, const double* const r, const int arr_size, const double alpha,
    const ProcComputingCoords procCoords){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; true; k += threadNum) {

        int i = static_cast<int>(procCoords.left) + k % (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
        
        int j = static_cast<int>(procCoords.top) + k / (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

        if (j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom)) {
            g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i] - alpha * g[j * procCoords.x_cells_num + i];
        } else {
            break;
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::cuda_Compute_g
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Compute_g (double* const g, const double* const r, const double alpha) const{

    // internal region
    int dimension = (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top)) *
        (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
    GridDistribute mesh (devProp, dimension);
    cudakernel_Compute_g<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>> (g, r, dimension, alpha, procCoords); CUDA_CHECK_LAST_ERROR;

    cudaAllStreamsSynchronize(0, 0);
}


// ==================================================================================================================================================
//                                                                                                                               cudakernel_Compute_p
// ==================================================================================================================================================
__global__ void cudakernel_Compute_p (double* const p, const double* const p_prev, const double* const g,  const int arr_size, const double tau,
    const ProcComputingCoords procCoords){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; true; k += threadNum) {

        int i = static_cast<int>(procCoords.left) + k % (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
        
        int j = static_cast<int>(procCoords.top) + k / (
            procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

        if (j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom)) {
            p[j * procCoords.x_cells_num + i] = p_prev[j * procCoords.x_cells_num + i] - tau * g[j * procCoords.x_cells_num + i];
        } else {
            break;
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::cuda_Compute_p
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Compute_p (const double tau, const double* const g) {
    
    int dimension = (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top)) *
        (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
    GridDistribute mesh (devProp, dimension);
    cudakernel_Compute_p<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>> (p, p_prev, g, dimension, tau, procCoords); CUDA_CHECK_LAST_ERROR;

    cudaAllStreamsSynchronize(0, 0);
}
