#include <utility>

using std::pair;

#include "DHP_PE_RA_FDM.h"

// ==================================================================================================================================================
//                                                                                                                 cudakernel_Counting_5_star_insides
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_insides (double* const delta_f, const double* const f, const ProcComputingCoords procCoords,
    const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;
    int i = 1+ threadLinearIdx % (procCoords.x_cells_num -2);
    int j = 1+ threadLinearIdx / (procCoords.x_cells_num -2);

    if (j < procCoords.y_cells_num -1)
        delta_f[j * procCoords.x_cells_num + i] = (
                (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
            ) / hx + (
                (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
            ) / hy;
}


// ==================================================================================================================================================
//                                                                                                 cudakernel_Counting_5_star_Memcpy_vertical_message
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_Memcpy_vertical_message (double* const to, const double* const from, const int elem_num, const int step){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    if (threadLinearIdx < elem_num)
        to[threadLinearIdx] = from[threadLinearIdx * step];
}


// ==================================================================================================================================================
//                                                                                                              cudakernel_Counting_5_star_LR_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_LR_delta_f (double* const delta_f, const double* const f, const double* const recv_message_lr,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = 0;
    int j = 1 + threadLinearIdx;

    if (j < procCoords.y_cells_num -1)
        if (not procCoords.left) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]               ) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                              cudakernel_Counting_5_star_RL_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_RL_delta_f (double* const delta_f, const double* const f, const double* const recv_message_rl,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = procCoords.x_cells_num -1;
    int j = 1+ threadLinearIdx;

    if (j < procCoords.y_cells_num -1)
        if (not procCoords.right) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                              cudakernel_Counting_5_star_TD_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_TD_delta_f (double* const delta_f, const double* const f, const double* const recv_message_td,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = 1 + threadLinearIdx;
    int j = 0;

    if (i < procCoords.x_cells_num -1)
        if (not procCoords.top) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td[i]               ) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[j * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                              cudakernel_Counting_5_star_BU_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_BU_delta_f (double* const delta_f, const double* const f, const double* const recv_message_bu,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = 1 + threadLinearIdx;
    int j = procCoords.y_cells_num -1;

    if (i < procCoords.x_cells_num -1)
        if (not procCoords.bottom) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (recv_message_bu[i]                    - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                            cudakernel_Counting_5_star_TDBU_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_TDBU_delta_f (double* const delta_f, const double* const f,
    const double* const recv_message_td, const double* const recv_message_bu,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = 1 + threadLinearIdx;
    int j = 0;

    if (i < procCoords.x_cells_num -1)
        if (not procCoords.top and not procCoords.bottom) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td[i]               ) / hy -
                    (recv_message_bu[i]                    - f[j * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                            cudakernel_Counting_5_star_LRRL_delta_f
// ==================================================================================================================================================
__global__ void cudakernel_Counting_5_star_LRRL_delta_f (double* const delta_f, const double* const f,
    const double* const recv_message_lr, const double* const recv_message_rl,
    const ProcComputingCoords procCoords, const int hx, const int hy){

    int threadLinearIdx = (threadIdx.z * blockDim.y * gridDim.y + threadIdx.y) * blockDim.y * gridDim.y + threadIdx.x;

    int i = 0;
    int j = 1+ threadLinearIdx;

    if (j < procCoords.y_cells_num -1)
        if (not procCoords.left and not procCoords.right) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]               ) / hx -
                    (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
}


// ==================================================================================================================================================
//                                                                                                                DHP_PE_RA_FDM::cuda_Counting_5_star
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cuda_Counting_5_star (const double* const f, double* const delta_f){

    int i = 0;
    int j = 0;
    int ret = MPI_SUCCESS;
    pair<dim3, dim3> mesh = GridDistribute((procCoords.x_cells_num -2) * (procCoords.y_cells_num -2));

    cudakernel_Counting_5_star_insides<<<mesh.first, mesh.second, 0, cudaStreams[0]>>> (delta_f, f, procCoords, hx, hy);

    // ==========================================
    // memory allocation
    // ==========================================

    if (send_message_lr == NULL)
        SAFE_CUDA(cudaHostAlloc(&send_message_lr, procCoords.y_cells_num * sizeof(*send_message_lr), cudaHostAllocMapped));
    if (send_message_rl == NULL)
        SAFE_CUDA(cudaHostAlloc(&send_message_rl, procCoords.y_cells_num * sizeof(*send_message_rl), cudaHostAllocMapped));
    if (send_message_td == NULL)
        SAFE_CUDA(cudaHostAlloc(&send_message_td, procCoords.x_cells_num * sizeof(*send_message_td), cudaHostAllocMapped));
    if (send_message_bu == NULL)
        SAFE_CUDA(cudaHostAlloc(&send_message_bu, procCoords.x_cells_num * sizeof(*send_message_bu), cudaHostAllocMapped));
    if (recv_message_lr == NULL)
        SAFE_CUDA(cudaHostAlloc(&recv_message_lr, procCoords.y_cells_num * sizeof(*recv_message_lr), cudaHostAllocMapped));
    if (recv_message_rl == NULL)
        SAFE_CUDA(cudaHostAlloc(&recv_message_rl, procCoords.y_cells_num * sizeof(*recv_message_rl), cudaHostAllocMapped));
    if (recv_message_td == NULL)
        SAFE_CUDA(cudaHostAlloc(&recv_message_td, procCoords.x_cells_num * sizeof(*recv_message_td), cudaHostAllocMapped));
    if (recv_message_bu == NULL)
        SAFE_CUDA(cudaHostAlloc(&recv_message_bu, procCoords.x_cells_num * sizeof(*recv_message_bu), cudaHostAllocMapped));

    // ==========================================
    // initialize send buffers
    // ==========================================

    mesh = GridDistribute(procCoords.y_cells_num);

    // left -> right
    cudakernel_Counting_5_star_Memcpy_vertical_message<<<mesh.first, mesh.second, 0 cudaStreams[1]>>> (send_message_lr, f + procCoords.x_cells_num -1,
        procCoords.y_cells_num, procCoords.x_cells_num);
    // right -> left
    cudakernel_Counting_5_star_Memcpy_vertical_message<<<mesh.first, mesh.second, 0 cudaStreams[2]>>> (send_message_rl, f,
        procCoords.y_cells_num, procCoords.x_cells_num);
    // top -> down
    SAFE_CUDA(cudaMemcpyAsync(send_message_td, f + (procCoords.y_cells_num -1) * procCoords.x_cells_num, procCoords.x_cells_num * sizeof(*f),
        cudaMemcpyDeviceToHost, cudaStreams[3]));
    // bottom -> up
    SAFE_CUDA(cudaMemcpyAsync(send_message_bu, f, procCoords.x_cells_num * sizeof(*f), cudaMemcpyDeviceToHost, cudaStreams[4]));


    int send_amount = 0;
    int recv_amount = 0;

    // ==========================================
    // send messages
    // ==========================================

    // left -> right
    if (not procCoords.right){

        cudaStreamSynchronize(cudaStreams[1]);
        ret = MPI_Isend(
            send_message_lr,                            // void* buffer
            procCoords.y_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank +1,                         // int dest
            DHP_PE_RA_FDM::StarLeftRight,               // int tag
            procParams.comm,                            // MPI_Comm comm
            &(send_reqs_5_star[send_amount])            // MPI_Request *request
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message from left to right.");
    }
    // right -> left
    if (not procCoords.left){

        cudaStreamSynchronize(cudaStreams[2]);
        ret = MPI_Isend(
            send_message_rl,                            // void* buffer
            procCoords.y_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank -1,                         // int dest
            DHP_PE_RA_FDM::StarRightLeft,               // int tag
            procParams.comm,                            // MPI_Comm comm
            &(send_reqs_5_star[send_amount])            // MPI_Request *request
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message from right to left.");
    }
    // top -> down
    if (not procCoords.bottom){

        cudaStreamSynchronize(cudaStreams[3]);
        ret = MPI_Isend(
            send_message_td,                            // void* buffer
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank + procCoords.x_proc_num,    // int dest
            DHP_PE_RA_FDM::StarTopDown,                 // int tag
            procParams.comm,                            // MPI_Comm comm
            &(send_reqs_5_star[send_amount])            // MPI_Request *request
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message top -> down.");
    }
    // bottom -> up
    if (not procCoords.top){

        cudaStreamSynchronize(cudaStreams[4]);
        ret = MPI_Isend(
            send_message_bu,                            // void* buffer
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank - procCoords.x_proc_num,    // int dest
            DHP_PE_RA_FDM::StarBottomUp,                // int tag
            procParams.comm,                            // MPI_Comm comm
            &(send_reqs_5_star[send_amount])            // MPI_Request *request
        );
        send_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message bottom -> up.");
    }

    // ==========================================
    // receive messages
    // ==========================================

    // left -> right
    if (not procCoords.left){

        ret = MPI_Irecv(
            recv_message_lr,                            // void *buf
            procCoords.y_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank -1,                         // int source
            DHP_PE_RA_FDM::StarLeftRight,               // int tag
            procParams.comm,                            // MPI_Comm comm
            &(recv_reqs_5_star[recv_amount])            // MPI_Request *request
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message from left to right.");
    }
    // right -> left
    if (not procCoords.right){

        ret = MPI_Irecv(
            recv_message_rl,                            // void *buf
            procCoords.y_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank +1,                         // int source
            DHP_PE_RA_FDM::StarRightLeft,               // int tag
            procParams.comm,                            // MPI_Comm comm
            &(recv_reqs_5_star[recv_amount])            // MPI_Request *request
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message from right to left.");
    }
    // top -> down
    if (not procCoords.top){

        ret = MPI_Irecv(
            recv_message_td,                            // void *buf
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank - procCoords.x_proc_num,    // int source
            DHP_PE_RA_FDM::StarTopDown,                 // int tag
            procParams.comm,                            // MPI_Comm comm
            &(recv_reqs_5_star[recv_amount])            // MPI_Request *request
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message top -> down.");
    }
    // bottom -> up
    if (not procCoords.bottom){

        ret = MPI_Irecv(
            recv_message_bu,                            // void *buf
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank + procCoords.x_proc_num,    // int source
            DHP_PE_RA_FDM::StarBottomUp,                // int tag
            procParams.comm,                            // MPI_Comm comm
            &(recv_reqs_5_star[recv_amount])            // MPI_Request *request
        );
        recv_amount++;

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message bottom -> up.");
    }

    // ==========================================
    // wait receiving all messages
    // ==========================================

    ret = MPI_Waitall(
        recv_amount,        // int count,
        recv_reqs_5_star,   // MPI_Request array_of_requests[],
        MPI_STATUS_IGNORE   // MPI_Status array_of_statuses[]
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error waiting for recv's in Counting_5_star.");

    // ==========================================
    // process received messages
    // ==========================================

    // Counting squared regions n x m, where n > 1 and m > 1
    if (procCoords.x_cells_num > 1 and procCoords.y_cells_num > 1)
    {
        // left -> right
        mesh = GridDistribute(procCoords.y_cells_num -2);
        cudakernel_Counting_5_star_LR_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[5]>>> (delta_f, f, recv_message_lr, procCoords, hx, hy);

        // right -> left
        mesh = GridDistribute(procCoords.y_cells_num -2);
        cudakernel_Counting_5_star_RL_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[6]>>> (delta_f, f, recv_message_rl, procCoords, hx, hy);

        // top -> down
        mesh = GridDistribute(procCoords.x_cells_num -2);
        cudakernel_Counting_5_star_TD_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[7]>>> (delta_f, f, recv_message_td, procCoords, hx, hy);

        // bottom -> up
        mesh = GridDistribute(procCoords.x_cells_num -2);
        cudakernel_Counting_5_star_BU_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[8]>>> (delta_f, f, recv_message_bu, procCoords, hx, hy);

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        j = 0;
        i = 0;
        if (not procCoords.top and not procCoords.left) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr [0] ) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td [0]                  ) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

        j = 0;
        i = procCoords.x_cells_num -1;
        if (not procCoords.top and not procCoords.right){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (recv_message_rl [0]                 - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td [procCoords.x_cells_num -1] ) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

        j = procCoords.y_cells_num -1;
        i = 0;
        if (not procCoords.bottom and not procCoords.left){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[procCoords.y_cells_num -1] ) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (recv_message_bu [0]                   - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

        j = procCoords.y_cells_num -1;
        i = procCoords.x_cells_num -1;
        if (not procCoords.bottom and not procCoords.right){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (recv_message_rl [procCoords.y_cells_num -1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (recv_message_bu [procCoords.x_cells_num -1] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

    } else if (procCoords.x_cells_num > 1 and procCoords.y_cells_num == 1){
        // Counting regions n x 1, where n > 1

        // top -> down
        // bottom -> up
        mesh = GridDistribute(procCoords.x_cells_num -2);
        cudakernel_Counting_5_star_TDBU_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[5]>>> (delta_f, f, recv_message_td, recv_message_bu, procCoords, hx, hy);

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        j = 0;
        i = 0;
        if (not procCoords.top and not procCoords.bottom and not procCoords.left) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr [0] ) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td [0]                  ) / hy -
                    (recv_message_bu[0]                    - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

        j = 0;
        i = procCoords.x_cells_num -1;
        if (not procCoords.top and not procCoords.bottom and not procCoords.right){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (recv_message_rl [0]                 - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td [procCoords.x_cells_num -1] ) / hy -
                    (recv_message_bu [procCoords.x_cells_num -1] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
            }

    } else if (procCoords.x_cells_num == 1 and procCoords.y_cells_num > 1){
        // Counting regions 1 x m, where m > 1

        // left -> right
        // right -> left
        mesh = GridDistribute(procCoords.y_cells_num -2);
        cudakernel_Counting_5_star_TDBU_delta_f<<<mesh.first, mesh.second, 0, cudaStreams[5]>>> (delta_f, f, recv_message_lr, recv_message_rl, procCoords, hx, hy);

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        j = 0;
        i = 0;
        if (not procCoords.left and not procCoords.right and not procCoords.top) {
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr [0]                ) / hx -
                    (recv_message_rl[0]                  - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td [0]                  ) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

        j = procCoords.y_cells_num -1;
        i = 0;
        if (not procCoords.left and not procCoords.right and not procCoords.bottom){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]                   ) / hx -
                    (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i  ]  ) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (recv_message_bu [0]                   - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }

    } else if (procCoords.x_cells_num == 1 and procCoords.y_cells_num == 1){
        // Counting regions 1 x 1
        
        i = 0;
        j = 0;
        if (not procCoords.left and not procCoords.right and not procCoords.top and not procCoords.bottom){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]                   ) / hx -
                    (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i  ]  ) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message_td[0]                   ) / hy -
                    (recv_message_bu [0]                   - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        } else {
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
    }

    // ==========================================
    // wait sending all messages
    // ==========================================

    ret = MPI_Waitall(
        send_amount,        // int count,
        send_reqs_5_star,   // MPI_Request array_of_requests[],
        MPI_STATUS_IGNORE   // MPI_Status array_of_statuses[]
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error waiting for sends after previous Counting_5_star.");

    cudaAllStreamsSynchronize(0, 8);
}
