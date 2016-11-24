#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                     cudakernel_Initialize_F_boundary_fi_horizontal
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_horizontal (double* const f, const ProcComputingCoords procCoords,
    const double X1, const double Y1, const double hx, const double hy, const int j_shift) {

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; k < procCoords.x_cells_num; k += threadNum)
        f[j_shift * procCoords.x_cells_num + k] = fi(
            X1 + (procCoords.x_cell_pos + k) * hx,
            Y1 + (procCoords.y_cell_pos + j_shift) * hy
        );
}


// ==================================================================================================================================================
//                                                                                                       cudakernel_Initialize_F_boundary_fi_vertical
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_vertical (double* const f, const ProcComputingCoords procCoords, 
    const double X1, const double Y1, const double hx, const double hy, const int i_shift) {

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; k < procCoords.y_cells_num; k += threadNum)
        f[k * procCoords.x_cells_num + i_shift] = fi(
            X1 + (procCoords.x_cell_pos + i_shift) * hx,
            Y1 + (procCoords.y_cell_pos + k) * hy
        );
}


// ==================================================================================================================================================
//                                                                                                         DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev (){

    // internal region
    int dimension = (procCoords.x_cells_num) * (procCoords.y_cells_num - static_cast<int>(procCoords.top) - static_cast<int>(procCoords.bottom));
    GridDistribute mesh (devProp, dimension);
    cudakernel_MemsetDouble<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>>
        (p_prev + static_cast<int>(procCoords.top) * procCoords.x_cells_num, 0, dimension); CUDA_CHECK_LAST_ERROR;

    // SAFE_CUDA(cudaMemsetAsync(p_prev + j * procCoords.x_cells_num + static_cast<int>(procCoords.left), 1,
    //     (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left)) * sizeof(*p_prev),
    //     cudaStreams[0]));
    cudaAllStreamsSynchronize(0, 0);


    mesh = GridDistribute (devProp, procCoords.x_cells_num);

    if (procCoords.top){

        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>>
            (p, procCoords, X1, Y1, hx, hy, 0); CUDA_CHECK_LAST_ERROR;
        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[4]>>>
            (p_prev, procCoords, X1, Y1, hx, hy, 0); CUDA_CHECK_LAST_ERROR;
    }
    if (procCoords.bottom){

        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[1]>>>
            (p, procCoords, X1, Y1, hx, hy, procCoords.y_cells_num -1); CUDA_CHECK_LAST_ERROR;
        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[5]>>>
            (p_prev, procCoords, X1, Y1, hx, hy, procCoords.y_cells_num -1); CUDA_CHECK_LAST_ERROR;
    }

    mesh = GridDistribute (devProp, procCoords.y_cells_num);

    if (procCoords.left){
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[2]>>>
            (p, procCoords, X1, Y1, hx, hy, 0); CUDA_CHECK_LAST_ERROR;
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[6]>>>
            (p_prev, procCoords, X1, Y1, hx, hy, 0); CUDA_CHECK_LAST_ERROR;
    }
    if (procCoords.right){
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[3]>>>
            (p, procCoords, X1, Y1, hx, hy, procCoords.x_cells_num -1); CUDA_CHECK_LAST_ERROR;
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[7]>>>
            (p_prev, procCoords, X1, Y1, hx, hy, procCoords.x_cells_num -1); CUDA_CHECK_LAST_ERROR;
    }

    cudaAllStreamsSynchronize(0, 7);
}


// ==================================================================================================================================================
//                                                                                                         cudakernel_Initialize_F_with_zero_vertical
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_with_zero_vertical (double* const f, const int elem_num, const int step){

    int threadId = THREAD_IN_GRID_ID;
    int threadNum = GRID_SIZE_IN_THREADS;

    for (int k = threadId; k < elem_num; k += threadNum)
        f[k * step] = 0;
}


// ==================================================================================================================================================
//                                                                                                  DHP_PE_RA_FDM::cuda_Initialize_F_border_with_zero
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Initialize_F_border_with_zero (double* const f){

    if (procCoords.top){

        int dimension = procCoords.x_cells_num;
        GridDistribute mesh (devProp, dimension);
        cudakernel_MemsetDouble<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[0]>>> (
            f + 0 * procCoords.x_cells_num + 0, 0, dimension); CUDA_CHECK_LAST_ERROR;

        // SAFE_CUDA(cudaMemset(f + 0 * procCoords.x_cells_num + 0, 0, procCoords.x_cells_num * sizeof(*f)));
    }
    if (procCoords.bottom){

        int dimension = procCoords.x_cells_num;
        GridDistribute mesh (devProp, dimension);
        cudakernel_MemsetDouble<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[1]>>> (
            f + (procCoords.y_cells_num -1) * procCoords.x_cells_num + 0, 0, dimension); CUDA_CHECK_LAST_ERROR;

        // SAFE_CUDA(cudaMemset(f + (procCoords.y_cells_num -1) * procCoords.x_cells_num + 0, 0, procCoords.x_cells_num * sizeof(*f)));
    }
    if (procCoords.left){

        GridDistribute mesh (devProp, procCoords.y_cells_num);
        cudakernel_Initialize_F_with_zero_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[2]>>> (
            f, procCoords.y_cells_num, procCoords.x_cells_num); CUDA_CHECK_LAST_ERROR;
    }
    if (procCoords.right){

        GridDistribute mesh (devProp, procCoords.y_cells_num);
        cudakernel_Initialize_F_with_zero_vertical<<<mesh.gridDim, mesh.blockDim, 0, cudaStreams[3]>>> (
            f + procCoords.x_cells_num -1, procCoords.y_cells_num, procCoords.x_cells_num); CUDA_CHECK_LAST_ERROR;
    }

    cudaAllStreamsSynchronize(0, 3);
}
