#include <algorithm>
#include <utility>

using std::min;
using std::pair;
using std::make_pair;

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda.h>

#include "DHP_PE_RA_FDM.h"

// #define fi(x, y) (x*x + y*y)/((1 + x*y)*(1 + x*y))
// #define F(x, y) logf(1 + x*y)

// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::GridDistribute
// ==================================================================================================================================================
pair<dim3, dim3> DHP_PE_RA_FDM::GridDistribute (const int demandedThreadNum){

    dim3 blockDim = dim3(min(min(demandedThreadNum, devProp.maxThreadsPerBlock), devProp.maxThreadsDim[0]));

    int demandedBlockNum = (demandedThreadNum -1) / blockDim.x +1;
    if (demandedBlockNum >= devProp.maxGridSize[0])
        throw DHP_PE_RA_FDM_Exception("Too many number of threads for device demanded.");

    dim3 gridDim = dim3(demandedBlockNum);
    
    return make_pair(gridDim, blockDim);
}


// ==================================================================================================================================================
//                                                                                                                    cukernel_Initialize_P_and_Pprev
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_top (double* const f, const ProcComputingCoords procCoords,
    const double X1, const double Y1, const double X2, const double Y2, const double hx, const double hy) {

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    if (threadLinearIdx < procCoords.x_cells_num)
        f[threadLinearIdx] = fi(X1 + (procCoords.x_cell_pos + threadLinearIdx) * hx, Y1);
}


// ==================================================================================================================================================
//                                                                                                                    cukernel_Initialize_P_and_Pprev
// ==================================================================================================================================================
__global__ void cudakernel_Initialize_F_boundary_fi_bottom (double* const f, const ProcComputingCoords procCoords,
    const double X1, const double Y1, const double X2, const double Y2, const double hx, const double hy) {

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    if (threadLinearIdx < procCoords.x_cells_num)
        f[(procCoords.y_cells_num -1) * procCoords.x_cells_num + threadLinearIdx] = fi(X1 + (procCoords.x_cell_pos + threadLinearIdx) * hx, Y2);
}


// ==================================================================================================================================================
//                                                                                                         DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Initialize_P_and_Pprev (){

    cudaStream_t stream1;
    cudaStream_t stream2;
    SAFE_CUDA(cudaStreamCreate(&stream1));
    SAFE_CUDA(cudaStreamCreate(&stream2));

    double* devicePtr1 = NULL;
    double* devicePtr2 = NULL;

    SAFE_CUDA(cudaMalloc(&devicePtr1, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr1)));
    SAFE_CUDA(cudaMemcpyAsync(devicePtr1, p, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr1), cudaMemcpyHostToDevice, stream1));

    SAFE_CUDA(cudaMalloc(&devicePtr2, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr2)));
    SAFE_CUDA(cudaMemcpyAsync(devicePtr2, p_prev, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr2), cudaMemcpyHostToDevice, stream2));


    if (procCoords.top){
        pair<dim3, dim3> mesh = GridDistribute(procCoords.x_cells_num);

        SAFE_CUDA(cudaStreamSynchronize(stream1));
        cudakernel_Initialize_F_boundary_fi_top<<<mesh.first, mesh.second, 0, stream1>>> (devicePtr1, procCoords, X1, Y1, X2, Y2, hx, hy);

        SAFE_CUDA(cudaStreamSynchronize(stream2));
        cudakernel_Initialize_F_boundary_fi_top<<<mesh.first, mesh.second, 0, stream2>>> (devicePtr2, procCoords, X1, Y1, X2, Y2, hx, hy);
    }

    SAFE_CUDA(cudaStreamSynchronize(stream1));
    SAFE_CUDA(cudaMemcpyAsync(p, devicePtr1, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr1), cudaMemcpyDeviceToHost, stream1));

    SAFE_CUDA(cudaStreamSynchronize(stream2));
    SAFE_CUDA(cudaMemcpyAsync(p_prev, devicePtr2, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*devicePtr2), cudaMemcpyDeviceToHost, stream2));

    SAFE_CUDA(cudaStreamSynchronize(stream1));
    SAFE_CUDA(cudaFree(devicePtr1));

    SAFE_CUDA(cudaStreamSynchronize(stream2));
    SAFE_CUDA(cudaFree(devicePtr2));

    SAFE_CUDA(cudaStreamDestroy(stream1));
    SAFE_CUDA(cudaStreamDestroy(stream2));


    // boundary region
    // if (procCoords.top){
    //     for (int i = 0; i < procCoords.x_cells_num; i++){
    //         p_prev[0 * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + 0) * hy);
    //         p[0 * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + 0) * hy);
    //     }
    // }
    if (procCoords.bottom){
        for (int i = 0; i < procCoords.x_cells_num; i++){
            p_prev[(procCoords.y_cells_num -1) * procCoords.x_cells_num + i] =
                fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + (procCoords.y_cells_num -1)) * hy);
            p[(procCoords.y_cells_num -1) * procCoords.x_cells_num + i] =
                fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + (procCoords.y_cells_num -1)) * hy);
        }
    }
    if (procCoords.left){
        for (int j = 0; j < procCoords.y_cells_num; j++){
            p_prev[j * procCoords.x_cells_num + 0] = fi(X1 + (procCoords.x_cell_pos + 0) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
            p[j * procCoords.x_cells_num + 0] = fi(X1 + (procCoords.x_cell_pos + 0) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
        }
    }
    if (procCoords.right){
        for (int j = 0; j < procCoords.y_cells_num; j++){
            p_prev[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] =
                fi(X1 + (procCoords.x_cell_pos + (procCoords.x_cells_num -1)) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
            p[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] =
                fi(X1 + (procCoords.x_cell_pos + (procCoords.x_cells_num -1)) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
        }
    }

    // internal region
    for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
        SAFE_CUDA(cudaMemset(p_prev + j * procCoords.x_cells_num + static_cast<int>(procCoords.left), 0,
            (procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left)) * sizeof(*p_prev)));
    }
}
