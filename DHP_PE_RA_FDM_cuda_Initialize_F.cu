#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                     cudakernel_Initialize_F_boundary_fi_horizontal
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_horizontal (double* const f, const ProcComputingCoords procCoords,
    const double X1, const double Y1, const double X2, const double Y2, const double hx, const double hy,
    const int j_shift) {

    int threadId = THREAD_IN_GRID_ID;

    if (threadId < procCoords.x_cells_num)
        f[j_shift * procCoords.x_cells_num + threadId] = fi(
            X1 + (procCoords.x_cell_pos + threadId) * hx,
            Y1 + (procCoords.y_cell_pos + j_shift) * hy
        );
}


// ==================================================================================================================================================
//                                                                                                       cudakernel_Initialize_F_boundary_fi_vertical
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_vertical (double* const f, const ProcComputingCoords procCoords, 
    const double X1, const double Y1, const double X2, const double Y2, const double hx, const double hy,
    const int i_shift) {

    int threadId = THREAD_IN_GRID_ID;

    if (threadId < procCoords.y_cells_num)
        f[threadId * procCoords.x_cells_num + i_shift] = fi(
            X1 + (procCoords.x_cell_pos + i_shift) * hx,
            Y1 + (procCoords.y_cell_pos + threadId) * hy
        );
}


// ==================================================================================================================================================
//                                                                                                         DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev (){

    pair<dim3, dim3> mesh = GridDistribute(procCoords.x_cells_num);
    
    if (procCoords.top){

        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.first, mesh.second, 0, cudaStreams[0]>>>
            (p, procCoords, X1, Y1, X2, Y2, hx, hy, 0);
        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.first, mesh.second, 0, cudaStreams[4]>>>
            (p_prev, procCoords, X1, Y1, X2, Y2, hx, hy, 0);
    }
    if (procCoords.bottom){

        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.first, mesh.second, 0, cudaStreams[1]>>>
            (p, procCoords, X1, Y1, X2, Y2, hx, hy, procCoords.y_cells_num -1);
        cudakernel_Initialize_F_boundary_fi_horizontal<<<mesh.first, mesh.second, 0, cudaStreams[5]>>>
            (p_prev, procCoords, X1, Y1, X2, Y2, hx, hy, procCoords.y_cells_num -1);
    }

    mesh = GridDistribute(procCoords.y_cells_num);

    if (procCoords.left){
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.first, mesh.second, 0, cudaStreams[2]>>>
            (p, procCoords, X1, Y1, X2, Y2, hx, hy, 0);
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.first, mesh.second, 0, cudaStreams[6]>>>
            (p_prev, procCoords, X1, Y1, X2, Y2, hx, hy, 0);
    }
    if (procCoords.right){
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.first, mesh.second, 0, cudaStreams[3]>>>
            (p, procCoords, X1, Y1, X2, Y2, hx, hy, procCoords.x_cells_num -1);
        cudakernel_Initialize_F_boundary_fi_vertical<<<mesh.first, mesh.second, 0, cudaStreams[7]>>>
            (p_prev, procCoords, X1, Y1, X2, Y2, hx, hy, procCoords.x_cells_num -1);
    }

    // internal region
    for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){

        int dimension = procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left);
        mesh = GridDistribute(dimension);
        cudakernel_MemsetDouble<<<mesh.first, mesh.second, 0, cudaStreams[8 + j % 5]>>>
            (p_prev + j * procCoords.x_cells_num + static_cast<int>(procCoords.left), 0, dimension);

        // SAFE_CUDA(cudaMemsetAsync(p_prev + j * procCoords.x_cells_num + static_cast<int>(procCoords.left), 1,
        //     (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left)) * sizeof(*p_prev),
        //     cudaStreams[8 + j % 5]));
    }

    cudaAllStreamsSynchronize(0, 13);
}


// ==================================================================================================================================================
//                                                                                                         cudakernel_Initialize_F_with_zero_vertical
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_with_zero_vertical (double* const f, const int elem_num, const int step){

    int threadId = THREAD_IN_GRID_ID;

    if (threadId < elem_num)
        f[threadId * step] = 0;
}


// ==================================================================================================================================================
//                                                                                                  DHP_PE_RA_FDM::cuda_Initialize_F_border_with_zero
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Initialize_F_border_with_zero (double* const f){

    if (procCoords.top){

        int dimension = procCoords.x_cells_num;
        pair<dim3, dim3> mesh = GridDistribute(dimension);
        cudakernel_MemsetDouble<<<mesh.first, mesh.second, 0, cudaStreams[0]>>> (f + 0 * procCoords.x_cells_num + 0, 0, dimension);

        // SAFE_CUDA(cudaMemset(f + 0 * procCoords.x_cells_num + 0, 0, procCoords.x_cells_num * sizeof(*f)));
    }
    if (procCoords.bottom){

        int dimension = procCoords.x_cells_num;
        pair<dim3, dim3> mesh = GridDistribute(dimension);
        cudakernel_MemsetDouble<<<mesh.first, mesh.second, 0, cudaStreams[1]>>> (
            f + (procCoords.y_cells_num -1) * procCoords.x_cells_num + 0, 0, dimension);

        // SAFE_CUDA(cudaMemset(f + (procCoords.y_cells_num -1) * procCoords.x_cells_num + 0, 0, procCoords.x_cells_num * sizeof(*f)));
    }
    if (procCoords.left){

        pair<dim3, dim3> mesh = GridDistribute(procCoords.y_cells_num);
        cudakernel_Initialize_F_with_zero_vertical<<<mesh.first, mesh.second, 0, cudaStreams[2]>>> (f, procCoords.y_cells_num, procCoords.x_cells_num);
    }
    if (procCoords.right){

        pair<dim3, dim3> mesh = GridDistribute(procCoords.y_cells_num);
        cudakernel_Initialize_F_with_zero_vertical<<<mesh.first, mesh.second, 0, cudaStreams[3]>>> (f + procCoords.x_cells_num -1, procCoords.y_cells_num, procCoords.x_cells_num);
    }

    cudaAllStreamsSynchronize(0, 3);
}
