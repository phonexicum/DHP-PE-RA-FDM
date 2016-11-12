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
//                                                                                                                               cudakernel_Compute_g
// ==================================================================================================================================================
__global__ void cudakernel_Compute_g (double* const g, const double* const r, const double alpha, const ProcComputingCoords procCoords){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = threadLinearIdx % (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
    int j = threadLinearIdx / (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

    g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i] - alpha * g[j * procCoords.x_cells_num + i];
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_g
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute_g (double* const g, const double* const r, const double alpha) const{

    // internal region
    pair<dim3, dim3> mesh = GridDistribute(
        (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top)) *
        (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left))
    );
    cudakernel_Compute_g<<<mesh.first, mesh.second, 0, cudaStreams[0]>>> (g, r, alpha, procCoords);

    cudaAllStreamsSynchronize(0, 0);
}


// ==================================================================================================================================================
//                                                                                                                               cudakernel_Compute_p
// ==================================================================================================================================================
__global__ void cudakernel_Compute_p (double* const p, const double* const p_prev, const double* const g, const double tau,
    const ProcComputingCoords procCoords){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = threadLinearIdx % (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));
    int j = threadLinearIdx / (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

    p[j * procCoords.x_cells_num + i] = p_prev[j * procCoords.x_cells_num + i] - tau * g[j * procCoords.x_cells_num + i];
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_p
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Compute_p (const double tau, const double* const g) {
    
    pair<dim3, dim3> mesh = GridDistribute(
        (procCoords.y_cells_num - static_cast<int>(procCoords.bottom) - static_cast<int>(procCoords.top)) *
        (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left))
    );
    cudakernel_Compute_g<<<mesh.first, mesh.second, 0, cudaStreams[0]>>> (p, p_prev, g, tau, procCoords);

    cudaAllStreamsSynchronize(0, 0);
}
