#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include <cmath>
#include <limits>
#include <iomanip>


#include <mpi.h>


#include "DHP_PE_RA_FDM.h"


using std::cout;
using std::fstream;
using std::endl;
using std::setw;

using std::ceil;
using std::max;
using std::abs;

using std::swap;

// ==================================================================================================================================================
//                                                                                                                       DHP_PE_RA_FDM::DHP_PE_RA_FDM
// ==================================================================================================================================================
ProcComputingCoords::ProcComputingCoords ():
x_proc_num (0),
y_proc_num (0),
x_cells_num (0),
x_cell_pos (0),
y_cells_num (0),
y_cell_pos (0),
top (false),
bottom (false),
left (false),
right (false)
{}


// ==================================================================================================================================================
//                                                                                                                       DHP_PE_RA_FDM::DHP_PE_RA_FDM
// ==================================================================================================================================================
ProcComputingCoords::ProcComputingCoords (const ProcParams& procParams, const int grid_size_x, const int grid_size_y, const int x_proc_num_, const int y_proc_num_){

    x_proc_num = x_proc_num_;
    y_proc_num = y_proc_num_;

    int x_cells_per_proc = (grid_size_x +1) / x_proc_num;
    int x_redundant_cells_num = (grid_size_x +1) % x_proc_num;
    int x_normal_tasks_num = x_proc_num - x_redundant_cells_num;

    if (procParams.rank % x_proc_num < x_normal_tasks_num) {
        x_cells_num = x_cells_per_proc;
        x_cell_pos = procParams.rank % x_proc_num * x_cells_per_proc;
    } else {
        x_cells_num = x_cells_per_proc + 1;
        x_cell_pos = procParams.rank % x_proc_num * x_cells_per_proc + (procParams.rank % x_proc_num - x_normal_tasks_num);
    }

    int y_cells_per_proc = (grid_size_y +1) / y_proc_num;
    int y_redundant_cells_num = (grid_size_y +1) % y_proc_num;
    int y_normal_tasks_num = y_proc_num - y_redundant_cells_num;

    if (procParams.rank / x_proc_num < y_normal_tasks_num) {
        y_cells_num = y_cells_per_proc;
        y_cell_pos = procParams.rank / x_proc_num * y_cells_per_proc;
    } else {
        y_cells_num = y_cells_per_proc + 1;
        y_cell_pos = procParams.rank / x_proc_num * y_cells_per_proc + (procParams.rank / x_proc_num - y_normal_tasks_num);
    }

    top = procParams.rank < x_proc_num;
    bottom = procParams.rank >= x_proc_num * (y_proc_num -1);
    left = procParams.rank % x_proc_num == 0;
    right = procParams.rank % x_proc_num == x_proc_num -1;
}


// ==================================================================================================================================================
//                                                                                                                       DHP_PE_RA_FDM::DHP_PE_RA_FDM
// ==================================================================================================================================================
DHP_PE_RA_FDM::DHP_PE_RA_FDM (  const double x1, const double y1, const double x2, const double y2, const int grid_size_x_, const int grid_size_y_,
                                const double eps_, const int cudaDeviceNum, const int descent_step_iterations_):
X1(x1), Y1(y1), 
X2(x2), Y2(y2),

hx ((x2-x1)/grid_size_x_),
hy ((y2-y1)/grid_size_y_),

grid_size_x (grid_size_x_),
grid_size_y (grid_size_y_),
eps (eps_),

local_f (NULL),

hxhy (hx*hy),
hx2 (hx*hx),
hy2 (hy*hy),

descent_step_iterations (descent_step_iterations_),
iterations_counter (0),

cuda_sum_aggr_arr1 (NULL),
cuda_sum_aggr_arr2 (NULL),

p (NULL),
p_prev (NULL),

send_message_lr (NULL),
send_message_rl (NULL),
send_message_td (NULL),
send_message_bu (NULL),
recv_message_lr (NULL),
recv_message_rl (NULL),
recv_message_td (NULL),
recv_message_bu (NULL),
recv_reqs_5_star (NULL),
send_reqs_5_star (NULL),

debug_fname (string("debug.txt"))

{
    // std::numeric_limits<double>::max_digits10 - is a c++11 feature
    // std::numeric_limits<double>::max_digits10 == 17
    cout << std::setprecision(17);

    send_reqs_5_star = new MPI_Request [4];
    recv_reqs_5_star = new MPI_Request [4];

    if (debug){
        int r;
        MPI_Comm_rank (MPI_COMM_WORLD, &r);
        if (r == 0){
            fstream fout (debug_fname.c_str());

            fout << "X1= " << X1 << " Y1= " << Y1 << endl
                << "X2= " << X2 << " Y2= " << Y2 << endl
                << "hx= " << hx << " hy= " << hy << endl
                << "Xgrid= " << grid_size_x << " Ygrid= " << grid_size_y << " eps= " << eps << endl;

            fout.close();
        }
    }

    // Setting memory mapping
    SAFE_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    SAFE_CUDA(cudaSetDeviceFlags(cudaDeviceBlockingSync));
    SAFE_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    SAFE_CUDA(cudaSetDevice(cudaDeviceNum));

    // Checking if device has necessary properties
    SAFE_CUDA(cudaGetDeviceProperties(&devProp, cudaDeviceNum));
    if (devProp.canMapHostMemory != 1 and devProp.unifiedAddressing != 1 and devProp.major < 2) {
        throw DHP_PE_RA_FDM_Exception ("Hardware does not fit this program. CUDA major version must be >= 2,"
                                       " memory must be able to be mapped and UAD must be enabled.");
    }

    for (int i = 0; i < cudaStreams_num; i++)
        SAFE_CUDA(cudaStreamCreate(cudaStreams + i));
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::~DHP_PE_RA_FDM
// ==================================================================================================================================================
DHP_PE_RA_FDM::~DHP_PE_RA_FDM (){

    if (local_f != NULL){
        SAFE_CUDA(cudaFreeHost(local_f)); local_f = NULL;
    }

    if (cuda_sum_aggr_arr1 != NULL){
        SAFE_CUDA(cudaFree(cuda_sum_aggr_arr1)); cuda_sum_aggr_arr1 = NULL;
    }
    if (cuda_sum_aggr_arr2 != NULL){
        SAFE_CUDA(cudaFree(cuda_sum_aggr_arr2)); cuda_sum_aggr_arr2 = NULL;
    }

    if (p != NULL){
        SAFE_CUDA(cudaFree(p)); p = NULL;
    }
    if (p_prev != NULL){
        SAFE_CUDA(cudaFree(p_prev)); p_prev = NULL;
    }

    if (send_message_lr != NULL){
        SAFE_CUDA(cudaFreeHost(send_message_lr)); send_message_lr = NULL;
    }
    if (send_message_rl != NULL){
        SAFE_CUDA(cudaFreeHost(send_message_rl)); send_message_rl = NULL;
    }
    if (send_message_td != NULL){
        SAFE_CUDA(cudaFreeHost(send_message_td)); send_message_td = NULL;
    }
    if (send_message_bu != NULL){
        SAFE_CUDA(cudaFreeHost(send_message_bu)); send_message_bu = NULL;
    }
    if (recv_message_lr != NULL){
        SAFE_CUDA(cudaFreeHost(recv_message_lr)); recv_message_lr = NULL;
    }
    if (recv_message_rl != NULL){
        SAFE_CUDA(cudaFreeHost(recv_message_rl)); recv_message_rl = NULL;
    }
    if (recv_message_td != NULL){
        SAFE_CUDA(cudaFreeHost(recv_message_td)); recv_message_td = NULL;
    }
    if (recv_message_bu != NULL){
        SAFE_CUDA(cudaFreeHost(recv_message_bu)); recv_message_bu = NULL;
    }
    if (recv_reqs_5_star != NULL){
        delete [] recv_reqs_5_star; recv_reqs_5_star = NULL;
    }
    if (send_reqs_5_star != NULL){
        delete [] send_reqs_5_star; send_reqs_5_star = NULL;
    }
    if (procParams.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&procParams.comm);
    }

    for (int i = 0; i < cudaStreams_num; i++)
        SAFE_CUDA(cudaStreamDestroy(cudaStreams[i]));
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::PrepareMPIComm
// ==================================================================================================================================================
MPI_Comm DHP_PE_RA_FDM::PrepareMPIComm(const ProcParams& procParams_in, const int x_proc_num, const int y_proc_num) const{

    if (procParams_in.size < x_proc_num * y_proc_num)
        throw DHP_PE_RA_FDM_Exception("Not enough processes for requested computations.");

    if (procParams_in.size > (grid_size_x+1) * (grid_size_y+1))
        throw DHP_PE_RA_FDM_Exception("Can not scale computation of matrix to demanded amount of processes (amount of points in region < amount of processes).");

    MPI_Comm newComm;
    if (procParams_in.rank < x_proc_num * y_proc_num){
        MPI_Comm_split(MPI_COMM_WORLD, 1, procParams_in.rank, &newComm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, procParams_in.rank, &newComm);
    }

    return newComm;
}


// ==================================================================================================================================================
//                                                                                                                             DHP_PE_RA_FDM::Compute
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute (const ProcParams& procParams_in, const int x_proc_num, const int y_proc_num){

    MPI_Comm algComm = PrepareMPIComm(procParams_in, x_proc_num, y_proc_num);
    if (algComm == MPI_COMM_NULL)
        return;

    if (procParams.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&procParams.comm);
    }
    procParams = ProcParams(algComm);
    procCoords = ProcComputingCoords(procParams, grid_size_x, grid_size_y, x_proc_num, y_proc_num);

    if (p != NULL)
        SAFE_CUDA(cudaFreeHost(p)); p = NULL;
    if (p_prev != NULL)
        SAFE_CUDA(cudaFreeHost(p_prev)); p_prev = NULL;

    double* g = NULL;
    double* r = NULL;
    double* delta_p = NULL;
    double* delta_g = NULL;
    double* delta_r = NULL;
    
    SAFE_CUDA(cudaMalloc(&p,       procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*p)));
    SAFE_CUDA(cudaMalloc(&p_prev,  procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*p_prev)));
    SAFE_CUDA(cudaMalloc(&g,       procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*g)));
    SAFE_CUDA(cudaMalloc(&r,       procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*r)));
    SAFE_CUDA(cudaMalloc(&delta_p, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*delta_p)));
    SAFE_CUDA(cudaMalloc(&delta_g, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*delta_g)));
    SAFE_CUDA(cudaMalloc(&delta_r, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*delta_r)));

    double scalar_product_delta_g_and_g = 1;
    double scalar_product_delta_r_and_g = 1;
    double scalar_product_r_and_g = 1;
    double alpha = 0;
    double tau = 0;

    // Computing step 1
    // Initialize_P_and_Pprev();
    cuda_Initialize_P_and_Pprev();
    Dump_func(debug_fname, p_prev, "p_prev");

    cuda_Initialize_F_border_with_zero(r);
    cuda_Initialize_F_border_with_zero(g);
    cuda_Initialize_F_border_with_zero(delta_p);
    cuda_Initialize_F_border_with_zero(delta_g);
    cuda_Initialize_F_border_with_zero(delta_r);


    iterations_counter = 0;
    while (true) {
        if (debug and procParams.rank == 0)
            cout << endl << "iterations_counter = " << iterations_counter << endl;

        // Computing step 2
        cuda_Counting_5_star(delta_p, p_prev);
        Dump_func(debug_fname, delta_p, "delta_p");

        // Computing step 3
        cuda_Compute_r (r, delta_p);
        Dump_func(debug_fname, r, "r");

        if (iterations_counter >= descent_step_iterations){
            // Computing step 4
            cuda_Counting_5_star(delta_r, r);
            Dump_func(debug_fname, delta_r, "delta_r");

            // Computing step 5
            scalar_product_delta_r_and_g = cuda_ComputingScalarProduct(delta_r, g);
            if (debug and procParams.rank == 0)
                cout << "scalar_product_delta_r_and_g= " << scalar_product_delta_r_and_g << endl;

            // Computing step 6
            alpha = scalar_product_delta_r_and_g / scalar_product_delta_g_and_g;
            if (debug and procParams.rank == 0)
                cout << "alpha= " << alpha << endl;
        }

        // Computing step 7
        if (iterations_counter >= descent_step_iterations){
            cuda_Compute_g (g, r, alpha);
            Dump_func(debug_fname, g, "g");
        } else {
            swap(g, r); // g is r now !
        }

        // Computing step 8
        cuda_Counting_5_star(delta_g, g);
        Dump_func(debug_fname, delta_g, "delta_g");

        // Computing step 9
        if (iterations_counter >= descent_step_iterations){
            scalar_product_r_and_g = cuda_ComputingScalarProduct(r, g);
            if (debug and procParams.rank == 0)
                cout << "scalar_product_r_and_g= " << scalar_product_r_and_g << endl;
        } else {
            scalar_product_r_and_g = cuda_ComputingScalarProduct(g, g); // because g is r now!
            if (debug and procParams.rank == 0)
                cout << "scalar_product_r_and_r= " << scalar_product_r_and_g << endl;
        }

        // Computing step 10
        scalar_product_delta_g_and_g = cuda_ComputingScalarProduct(delta_g, g);
        if (debug and procParams.rank == 0)
            cout << "scalar_product_delta_g_and_g= " << scalar_product_delta_g_and_g << endl;

        // Computing step 11
        if (scalar_product_delta_g_and_g != 0){
            tau = scalar_product_r_and_g / scalar_product_delta_g_and_g;
        } else {
            throw DHP_PE_RA_FDM_Exception( "Error. there is a divizion by zero in computations. Zero value is scalar product of delta_g and g.\n"
                "Probably there can be next problems:\n"
                "\t1) You have to increase amount of descent step iterations (default: 1);\n"
                "\t2) Algorithm has problems with fraction(fixed point part) of type 'double'.\n"
                "\t3) You specified too small matrix (internal region does not contain internal points, the only points are in boundary region).\n"
                "Iteration process terminated.");

            // 'p' - is side effect of computation and within error, last correct computation will be restored from p_prev
            swap(p, p_prev);
            break;
        }
        if (debug and procParams.rank == 0)
            cout << "tau= " << tau << endl;

        // Computing step 12
        cuda_Compute_p (tau, g);
        Dump_func(debug_fname, p, "p");

        // OutputBias (p_prev);
        if (cuda_StopCriteria (p, p_prev))
            break;

        swap(p, p_prev);
        iterations_counter++;
    }

    SAFE_CUDA(cudaFree(g)); g = NULL;
    SAFE_CUDA(cudaFree(r)); r = NULL;
    SAFE_CUDA(cudaFree(delta_p)); delta_p = NULL;
    SAFE_CUDA(cudaFree(delta_g)); delta_g = NULL;
    SAFE_CUDA(cudaFree(delta_r)); delta_r = NULL;
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Dump_func
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Dump_func(const string& fout_name, const double* const f, const string& func_label, const bool uncondutional){

    if (debug or uncondutional){

        if (local_f == NULL)
            SAFE_CUDA(cudaHostAlloc(&local_f, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*local_f), cudaHostAllocMapped));
            // SAFE_CUDA(cudaMallocHost(&local_f, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*local_f)));

        SAFE_CUDA(cudaMemcpy(local_f, f, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*f), cudaMemcpyDeviceToHost));

        if (procParams.rank != 0) {
            MPI_Status status;
            char fiction;

            MPI_Recv(
                &fiction,                   // void *buf
                1,                          // int count
                MPI_CHAR,                   // MPI_Datatype datatype
                procParams.rank -1,         // int source
                DHP_PE_RA_FDM::DumpSync,    // int tag
                procParams.comm,            // MPI_Comm comm
                &status                     // MPI_Status *status
            );
        }

        fstream fout (fout_name.c_str(), fstream::out | fstream::app);
        fout << "========== Dump begin ==========" << endl;

        fout << "procRank= " << procParams.rank << endl
            << "iterationsCounter= " << iterations_counter << endl << endl;

        if (f != NULL){
            fout << endl << func_label << " (x y f)" << endl;// << std::fixed;

            for (int j = 0; j < procCoords.y_cells_num; j++){
                for (int i = 0; i < procCoords.x_cells_num; i++){
                    
                    fout << std::setprecision(3)
                        << setw (4) << X1 + (procCoords.x_cell_pos + i) * hx << " "
                        << setw (4) << Y1 + (procCoords.y_cell_pos + j) * hy << " ";

                    // std::numeric_limits<double>::max_digits10 - is a c++11 feature
                    // std::numeric_limits<double>::max_digits10 == 17
                    fout << std::setprecision(17) << local_f[j * procCoords.x_cells_num + i] << " ";

                    fout << endl;

                }
                fout << endl;
            }
        }
        fout << endl;

        procCoords.Dump(fout_name);

        fout << "========== Dump end ==========" << endl;
        fout.close();

        if (procParams.rank != procParams.size -1){
            char fiction;

            MPI_Ssend(
                &fiction,                   // void* buffer
                1,                          // int count
                MPI_CHAR,                   // MPI_Datatype datatype
                procParams.rank +1,         // int dest
                DHP_PE_RA_FDM::DumpSync,    // int tag
                procParams.comm             // MPI_Comm comm
            );
        }

        MPI_Barrier (procParams.comm);
    }
}


// ==================================================================================================================================================
//                                                                                                                          ProcComputingCoords::Dump
// ==================================================================================================================================================
void ProcComputingCoords::Dump(const string& fout_name) const{
    fstream fout (fout_name.c_str(), fstream::out | fstream::app);

    fout << "x_proc_num= " << x_proc_num << " y_proc_num= " << y_proc_num << endl
        << "x_cells_num= " << x_cells_num << " x_cell_pos= " << x_cell_pos << endl
        << "y_cells_num= " << y_cells_num << " y_cell_pos= " << y_cell_pos << endl
        << "top= " << top << " bottom= " << bottom << " left= " << left << " right= " << right << endl;

    fout.close();
}


// ==================================================================================================================================================
//                                                                                                                          DHP_PE_RA_FDM::OutputBias
// ==================================================================================================================================================
// void DHP_PE_RA_FDM::OutputBias (const double* const f){
//     // This is DEBUG function only!

//     #define FI(x, y) (std::log(1 + x*y))

//     if (countBias){

//         double* p_dist = new double [procCoords.x_cells_num * procCoords.y_cells_num];

//         SAFE_CUDA(cudaMemcpy(p_dist, f, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*f), cudaMemcpyDeviceToHost));

//         double bias = 0;
//         for (int j = 0; j < procCoords.y_cells_num; j++)
//             for (int i = 0; i < procCoords.x_cells_num; i++){
//                 double val = p_dist[j * procCoords.x_cells_num + i] -
//                     FI(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
//                 bias += val * val * hxhy;
//             }

//         bias = std::sqrt(bias);

//         fstream fout ("bias.dat", fstream::out | fstream::app);
        
//         if (procParams.rank == 0)
//             // cout << "it= " << iterations_counter << " bias= " << bias << endl;
//             fout << iterations_counter << " " << bias << endl;

//         fout.close();

//         delete [] p_dist;
//     }
// }
