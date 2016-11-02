#include <iostream>
#include <fstream>
#include <string>

#include <cmath>
#include <limits>
#include <iomanip>

#include <mpi.h>

#include <unistd.h> // sleep


#include "DHP_PE_RA_FDM.h"


using std::cin;
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
ProcComputingCoords::ProcComputingCoords (const ProcParams& procParams, const uint grid_size_, const uint x_proc_num_, const uint y_proc_num_){

    x_proc_num = x_proc_num_;
    y_proc_num = y_proc_num_;

    uint x_cells_per_proc = (grid_size_ +1) / x_proc_num;
    uint x_redundant_cells_num = (grid_size_ +1) % x_proc_num;
    uint x_normal_tasks_num = x_proc_num - x_redundant_cells_num;

    if (procParams.rank % x_proc_num < x_normal_tasks_num) {
        x_cells_num = x_cells_per_proc;
        x_cell_pos = procParams.rank % x_proc_num * x_cells_per_proc;
    } else {
        x_cells_num = x_cells_per_proc + 1;
        x_cell_pos = procParams.rank % x_proc_num * x_cells_per_proc + (procParams.rank % x_proc_num - x_normal_tasks_num);
    }

    uint y_cells_per_proc = (grid_size_ +1) / y_proc_num;
    uint y_redundant_cells_num = (grid_size_ +1) % y_proc_num;
    uint y_normal_tasks_num = y_proc_num - y_redundant_cells_num;

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
DHP_PE_RA_FDM::DHP_PE_RA_FDM (  const double x1, const double y1, const double x2, const double y2, const uint grid_size_, const double eps_,
                                const uint descent_step_iterations_):
X1(x1), Y1(y1), 
X2(x2), Y2(y2),

hx ((x2-x1)/grid_size_),
hy ((y2-y1)/grid_size_),

grid_size (grid_size_),
eps (eps_),
descent_step_iterations (descent_step_iterations_),
iterations_counter (0),

p (nullptr),
p_prev (nullptr),
send_message (nullptr),
recv_message (nullptr),
gather_double_per_process (nullptr),

debug_fname (string("debug.txt"))
{
    cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    if (debug){
        int r;
        MPI_Comm_rank (MPI_COMM_WORLD, &r);
        if (r == 0){
            fstream fout (debug_fname);

            fout << "X1= " << X1 << " Y1= " << Y1 << endl
                << "X2= " << X2 << " Y2= " << Y2 << endl
                << "hx= " << hx << " hy= " << hy << endl
                << "grid= " << grid_size << " eps= " << eps << endl;

            fout.close();
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::~DHP_PE_RA_FDM
// ==================================================================================================================================================
DHP_PE_RA_FDM::~DHP_PE_RA_FDM (){
    if (p != nullptr){
        delete [] p; p = nullptr;
    }
    if (p_prev != nullptr){
        delete [] p_prev; p_prev = nullptr;
    }
    if (send_message != nullptr){
        delete [] send_message; send_message = nullptr;
    }
    if (recv_message != nullptr){
        delete [] recv_message; recv_message = nullptr;
    }
    if (gather_double_per_process != nullptr){
        delete [] gather_double_per_process; gather_double_per_process = nullptr;
    }
}


// ==================================================================================================================================================
//                                                                                                                             DHP_PE_RA_FDM::Compute
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute (const ProcParams& procParams, const uint x_proc_num, const uint y_proc_num){

    ProcComputingCoords procCoords (procParams, grid_size, x_proc_num, y_proc_num);

    if (p != nullptr)
        delete [] p; p = nullptr;
    if (p_prev != nullptr)
        delete [] p_prev; p_prev = nullptr;

    p = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    p_prev = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* g = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* r = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_p = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_g = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_r = new double [procCoords.x_cells_num * procCoords.y_cells_num];

    double scalar_product_delta_g_and_g = 1;

    // Computing step 1
    for (uint j = 0; j < procCoords.y_cells_num; j++){
        for (uint i = 0; i < procCoords.x_cells_num; i++){
            if ((procCoords.left and i == 0)                            or
                (procCoords.right and i == procCoords.x_cells_num -1)   or
                (procCoords.top and j == 0)                             or
                (procCoords.bottom and j == procCoords.y_cells_num -1)  )
            {
                p_prev[j * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
                p[j * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
            } else {
                p_prev[j * procCoords.x_cells_num + i] = 0;
            }
        }
    }
    Dump_func(debug_fname, procParams, procCoords, p_prev, "p_prev");


    iterations_counter = 0;
    while (true) {
        if (debug and procParams.rank == 0)
            cout << endl << "iterations_counter = " << iterations_counter << endl;

        // Computing step 2
        Counting_5_star (p_prev, delta_p, procParams, procCoords);
        Dump_func(debug_fname, procParams, procCoords, delta_p, "delta_p");

        // Computing step 3
        for (uint j = 0; j < procCoords.y_cells_num; j++){
            for (uint i = 0; i < procCoords.x_cells_num; i++){
                if ((procCoords.left and i == 0)                            or
                    (procCoords.right and i == procCoords.x_cells_num -1)   or
                    (procCoords.top and j == 0)                             or
                    (procCoords.bottom and j == procCoords.y_cells_num -1)  )
                {
                    r[j * procCoords.x_cells_num + i] = 0;
                } else {
                    r[j * procCoords.x_cells_num + i] =
                        delta_p[j * procCoords.x_cells_num + i] -
                        F(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy)
                        ;
                }
            }
        }
        Dump_func(debug_fname, procParams, procCoords, r, "r");

        double scalar_product_delta_r_and_g = 1;
        if (iterations_counter >= descent_step_iterations){
            // Computing step 4
            Counting_5_star (r, delta_r, procParams, procCoords);
            Dump_func(debug_fname, procParams, procCoords, delta_r, "delta_r");

            // Computing step 5
            scalar_product_delta_r_and_g = ComputingScalarProduct(delta_r, g, procParams, procCoords);
            if (debug and procParams.rank == 0)
                cout << "scalar_product_delta_r_and_g= " << scalar_product_delta_r_and_g << endl;
        }

        // Computing step 6
        double alpha_k = 0;
        if (iterations_counter >= descent_step_iterations){
            if (procParams.rank == 0){
                alpha_k = scalar_product_delta_r_and_g / scalar_product_delta_g_and_g;
            }
            alpha_k = BroadcastParameter(alpha_k, procParams);
            if (debug and procParams.rank == 0)
                cout << "alpha_k= " << alpha_k << endl;
        }

        // Computing step 7
        for (uint j = 0; j < procCoords.y_cells_num; j++){
            for (uint i = 0; i < procCoords.x_cells_num; i++){
                if ((procCoords.left and i == 0)                            or
                    (procCoords.right and i == procCoords.x_cells_num -1)   or
                    (procCoords.top and j == 0)                             or
                    (procCoords.bottom and j == procCoords.y_cells_num -1)  )
                {
                    g[j * procCoords.x_cells_num + i] = 0;
                } else {
                    if (iterations_counter >= descent_step_iterations){
                        g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i] - alpha_k * g[j * procCoords.x_cells_num + i];
                    } else {
                        g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i];
                    }
                }
            }
        }
        Dump_func(debug_fname, procParams, procCoords, g, "g");

        // Computing step 8
        Counting_5_star (g, delta_g, procParams, procCoords);
        Dump_func(debug_fname, procParams, procCoords, delta_g, "delta_g");

        // Computing step 9
        double scalar_product_r_and_g = ComputingScalarProduct(r, g, procParams, procCoords);
        if (debug and procParams.rank == 0)
            cout << "scalar_product_r_and_g= " << scalar_product_r_and_g << endl;

        // Computing step 10
        scalar_product_delta_g_and_g = ComputingScalarProduct(delta_g, g, procParams, procCoords);
        if (debug and procParams.rank == 0)
            cout << "scalar_product_delta_g_and_g= " << scalar_product_delta_g_and_g << endl;

        // Computing step 11
        double tau = 0;
        if (procParams.rank == 0){
            if (scalar_product_delta_g_and_g != 0){
                tau = scalar_product_r_and_g / scalar_product_delta_g_and_g;
            } else {
                // tau can not be negative by algorithm,
                //  this is a sign for error catching
                tau = -1;
            }
        }
        tau = BroadcastParameter(tau, procParams);
        if (debug and procParams.rank == 0)
            cout << "tau= " << tau << endl;

        if (tau == -1){
            if (procParams.rank == 0)
                cout << endl << "Error. there is a divizion by zero in computations. Zero value is scalar product of delta_g and g. " << endl
                    << "Probably there can be next problems:" << endl
                    << "\t1) You have to increase amount of descent step iterations (default: 1);" << endl
                    << "\t2) Algorithm has problems with fraction(fixed point part) of type 'double'." << endl
                    << "Iteration process stopped, you can examine result of unfinished computations (the result can be pretty rough)." << endl << endl;

            // 'p' - is side effect of computation and within error,
            // last correct computation will be taken from p_prev
            swap(p, p_prev);
            break;
        }

        // Computing step 12
        for (uint j = 0; j < procCoords.y_cells_num; j++){
            for (uint i = 0; i < procCoords.x_cells_num; i++){
                if ((procCoords.left and i == 0)                            or
                    (procCoords.right and i == procCoords.x_cells_num -1)   or
                    (procCoords.top and j == 0)                             or
                    (procCoords.bottom and j == procCoords.y_cells_num -1)  )
                {} else {
                    p[j * procCoords.x_cells_num + i] = p_prev[j * procCoords.x_cells_num + i] - tau * g[j * procCoords.x_cells_num + i];
                }
            }
        }
        Dump_func(debug_fname, procParams, procCoords, p, "p");

        if (stopCriteria (p, p_prev, procParams, procCoords))
            break;

        swap(p, p_prev);
        iterations_counter++;
    }

    delete [] g; g = nullptr;
    delete [] r; r = nullptr;
    delete [] delta_p; delta_p = nullptr;
    delete [] delta_g; delta_g = nullptr;
    delete [] delta_r; delta_r = nullptr;
}

// ==================================================================================================================================================
//                                                                                                                  DHP_PE_RA_FDM::BroadcastParameter
// ==================================================================================================================================================
double DHP_PE_RA_FDM::BroadcastParameter (double param, const ProcParams& procParams){

    int ret = MPI_Bcast(
        &param,         // void *buffer,
        1,              // int count,
        MPI_DOUBLE,     // MPI_Datatype datatype,
        0,              // int root, 
        procParams.comm // MPI_Comm comm
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error broadcasting value.");

    return param;
}


// ==================================================================================================================================================
//                                                                                                              DHP_PE_RA_FDM::ComputingScalarProduct
// ==================================================================================================================================================
double DHP_PE_RA_FDM::ComputingScalarProduct (  const double* const f, const double* const delta_f,
                                                const ProcParams& procParams, const ProcComputingCoords& procCoords){

    double scalar_product = 0;
    for (uint j = 0; j < procCoords.y_cells_num; j++){
        for (uint i = 0; i < procCoords.x_cells_num; i++){
            scalar_product += hx * hy * f[j * procCoords.x_cells_num + i] * delta_f[j * procCoords.x_cells_num + i];
        }
    }

    if (gather_double_per_process == nullptr and procParams.rank == 0)
        gather_double_per_process = new double [procParams.size];

    int ret = MPI_Gather(
        &scalar_product,            // const void *sendbuf,
        1,                          // int sendcount,
        MPI_DOUBLE,                 // MPI_Datatype sendtype,
        gather_double_per_process,  // void *recvbuf,
        1,                          // int recvcount, (per process !)
        MPI_DOUBLE,                 // MPI_Datatype recvtype,
        0,                          // int root,
        procParams.comm             // MPI_Comm comm
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error gathering scalar_product.");

    if (procParams.rank == 0){
        scalar_product = 0;
        for (uint i = 0; i < procParams.size; i++){
            scalar_product += gather_double_per_process[i];
        }
        return scalar_product;
    } else {
        return 0;
    }
}


// ==================================================================================================================================================
//                                                                                                                     DHP_PE_RA_FDM::Counting_5_star
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Counting_5_star (const double* const f, double* const delta_f, const ProcParams& procParams, const ProcComputingCoords& procCoords){

    uint i = 0;
    uint j = 0;

    for (j = 1; j < procCoords.y_cells_num -1; j++){
        for (i = 1; i < procCoords.x_cells_num -1; i++){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        }
    }

    if (send_message == nullptr)
        send_message = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (recv_message == nullptr)
        recv_message = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];

    double top_left_left = 0;
    double top_left_up = 0;
    double top_right_right = 0;
    double top_right_up = 0;
    double bottom_left_left = 0;
    double bottom_left_down = 0;
    double bottom_right_right = 0;
    double bottom_right_down = 0;

    // ================================
    // left -> right
    // ================================

    for (i = 0; i < procCoords.y_cells_num; i++){
        send_message[i] = p[ (i+1) * procCoords.x_cells_num -1];
    }

    if (not procCoords.right){

        int ret = MPI_Ssend(
            send_message,                    // void* buffer
            procCoords.y_cells_num,          // int count
            MPI_DOUBLE,                      // MPI_Datatype datatype
            procParams.rank +1,              // int dest
            MPI_MessageTypes::StarLeftRight, // int tag
            procParams.comm                  // MPI_Comm comm
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message from left to right.");
    }
    if (not procCoords.left){

        MPI_Status status;

        int ret = MPI_Recv(
            recv_message,                    // void *buf
            procCoords.y_cells_num,          // int count
            MPI_DOUBLE,                      // MPI_Datatype datatype
            procParams.rank -1,              // int source
            MPI_MessageTypes::StarLeftRight, // int tag
            procParams.comm,                 // MPI_Comm comm
            &status                          // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message from left to right.");
    }

    if (not procCoords.left and not procCoords.right){

        MPI_Status status;

        int ret = MPI_Sendrecv(
            send_message,                       // const void *sendbuf,
            procCoords.y_cells_num,             // int sendcount,
            MPI_DOUBLE,                         // MPI_Datatype sendtype,
            procParams.rank +1,                 // int dest,
            MPI_MessageTypes::StarLeftRight,    // int sendtag,
            recv_message,                       // void *recvbuf,
            procCoords.y_cells_num,             // int recvcount,
            MPI_DOUBLE,                         // MPI_Datatype recvtype,
            procParams.rank -1,                 // int source,
            MPI_MessageTypes::StarLeftRight,    // int recvtag,
            procParams.comm,                    // MPI_Comm comm,
            &status                             // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sendrecv message from left to right.");
    }

    i = 0;
    if (not procCoords.left and not (procCoords.right and i == procCoords.x_cells_num -1)) {
        top_left_left = recv_message [0];
        bottom_left_left = recv_message [procCoords.y_cells_num -1];

        for (j = 1; j < procCoords.y_cells_num -1; j++){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - recv_message[j]                  ) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        }
    } else {
        for (j = 1; j < procCoords.y_cells_num -1; j++){
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
    }


    // ================================
    // right -> left
    // ================================

    for (i = 0; i < procCoords.y_cells_num; i++){
        send_message[i] = p[ i * procCoords.x_cells_num + 0];
    }

    if (not procCoords.left){

        int ret = MPI_Ssend(
            send_message,                    // void* buffer
            procCoords.y_cells_num,          // int count
            MPI_DOUBLE,                      // MPI_Datatype datatype
            procParams.rank -1,              // int dest
            MPI_MessageTypes::StarRightLeft, // int tag
            procParams.comm                  // MPI_Comm comm
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message from right to left.");
    }
    if (not procCoords.right){

        MPI_Status status;

        int ret = MPI_Recv(
            recv_message,                    // void *buf
            procCoords.y_cells_num,          // int count
            MPI_DOUBLE,                      // MPI_Datatype datatype
            procParams.rank +1,              // int source
            MPI_MessageTypes::StarRightLeft, // int tag
            procParams.comm,                 // MPI_Comm comm
            &status                          // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message from right to left.");
    }

    if (not procCoords.left and not procCoords.right){

        MPI_Status status;

        int ret = MPI_Sendrecv(
            send_message,                       // const void *sendbuf,
            procCoords.y_cells_num,             // int sendcount,
            MPI_DOUBLE,                         // MPI_Datatype sendtype,
            procParams.rank -1,                 // int dest,
            MPI_MessageTypes::StarRightLeft,    // int sendtag,
            recv_message,                       // void *recvbuf,
            procCoords.y_cells_num,             // int recvcount,
            MPI_DOUBLE,                         // MPI_Datatype recvtype,
            procParams.rank +1,                 // int source,
            MPI_MessageTypes::StarRightLeft,    // int recvtag,
            procParams.comm,                    // MPI_Comm comm,
            &status                             // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sendrecv message from right to left.");
    }

    i = procCoords.x_cells_num -1;
    if (not procCoords.right and not (procCoords.left and i == 0)) {
        top_right_right = recv_message [0];
        bottom_right_right = recv_message [procCoords.y_cells_num -1];

        for (j = 1; j < procCoords.y_cells_num -1; j++){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (recv_message[j]                     - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        }
    } else {
        for (j = 1; j < procCoords.y_cells_num -1; j++){
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
    }


    // ================================
    // top -> down
    // ================================

    for (i = 0; i < procCoords.x_cells_num; i++){
        send_message[i] = p[ (procCoords.y_cell_pos + procCoords.y_cells_num -1) * procCoords.x_cells_num + i];
    }

    if (not procCoords.bottom){

        int ret = MPI_Ssend(
            send_message,                               // void* buffer
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank + procCoords.x_proc_num,    // int dest
            MPI_MessageTypes::StarTopDown,              // int tag
            procParams.comm                             // MPI_Comm comm
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message top -> down.");
    }
    if (not procCoords.top){

        MPI_Status status;

        int ret = MPI_Recv(
            recv_message,                                // void *buf
            procCoords.x_cells_num,                      // int count
            MPI_DOUBLE,                                  // MPI_Datatype datatype
            procParams.rank - procCoords.x_proc_num,     // int source
            MPI_MessageTypes::StarTopDown,               // int tag
            procParams.comm,                             // MPI_Comm comm
            &status                                      // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message top -> down.");
    }

    if (not procCoords.top and not procCoords.bottom){

        MPI_Status status;

        int ret = MPI_Sendrecv(
            send_message,                               // const void *sendbuf,
            procCoords.x_cells_num,                     // int sendcount,
            MPI_DOUBLE,                                 // MPI_Datatype sendtype,
            procParams.rank + procCoords.x_proc_num,    // int dest,
            MPI_MessageTypes::StarTopDown,              // int sendtag,
            recv_message,                               // void *recvbuf,
            procCoords.x_cells_num,                     // int recvcount,
            MPI_DOUBLE,                                 // MPI_Datatype recvtype,
            procParams.rank - procCoords.x_proc_num,    // int source,
            MPI_MessageTypes::StarTopDown,              // int recvtag,
            procParams.comm,                            // MPI_Comm comm,
            &status                                     // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sendrecv message top -> down.");
    }

    j = 0;
    if (not procCoords.top and not (procCoords.bottom and j == procCoords.y_cells_num -1)) {
        top_left_up = recv_message [0];
        top_right_up = recv_message [procCoords.x_cells_num -1];

        for (i = 1; i < procCoords.x_cells_num -1; i++){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - recv_message[i]                  ) / hy -
                    (f[(j+1) * procCoords.x_cells_num + i] - f[j * procCoords.x_cells_num + i]) / hy
                ) / hy;
        }
    } else {
        for (i = 1; i < procCoords.x_cells_num -1; i++){
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
    }


    // ================================
    // bottom -> up
    // ================================

    for (i = 0; i < procCoords.x_cells_num; i++){
        send_message[i] = p[i];
    }

    if (not procCoords.top){

        int ret = MPI_Ssend(
            send_message,                               // void* buffer
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank - procCoords.x_proc_num,    // int dest
            MPI_MessageTypes::StarBottomUp,             // int tag
            procParams.comm                             // MPI_Comm comm
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sending message bottom -> up.");
    }
    if (not procCoords.bottom){

        MPI_Status status;

        int ret = MPI_Recv(
            recv_message,                               // void *buf
            procCoords.x_cells_num,                     // int count
            MPI_DOUBLE,                                 // MPI_Datatype datatype
            procParams.rank + procCoords.x_proc_num,    // int source
            MPI_MessageTypes::StarBottomUp,             // int tag
            procParams.comm,                            // MPI_Comm comm
            &status                                     // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error receiving message bottom -> up.");
    }

    if (not procCoords.top and not procCoords.bottom){

        MPI_Status status;

        int ret = MPI_Sendrecv(
            send_message,                               // const void *sendbuf,
            procCoords.x_cells_num,                     // int sendcount,
            MPI_DOUBLE,                                 // MPI_Datatype sendtype,
            procParams.rank - procCoords.x_proc_num,    // int dest,
            MPI_MessageTypes::StarBottomUp,             // int sendtag,
            recv_message,                               // void *recvbuf,
            procCoords.x_cells_num,                     // int recvcount,
            MPI_DOUBLE,                                 // MPI_Datatype recvtype,
            procParams.rank + procCoords.x_proc_num,    // int source,
            MPI_MessageTypes::StarBottomUp,             // int recvtag,
            procParams.comm,                            // MPI_Comm comm,
            &status                                     // MPI_Status *status
        );

        if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error sendrecv message bottom -> up.");
    }

    j = procCoords.y_cells_num -1;
    if (not procCoords.bottom and not (procCoords.top and j == 0)) {
        bottom_left_down = recv_message [0];
        bottom_right_down = recv_message [procCoords.x_cells_num -1];

        for (i = 1; i < procCoords.x_cells_num -1; i++){
            delta_f[j * procCoords.x_cells_num + i] = (
                    (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                    (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                ) / hx + (
                    (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                    (recv_message[i]                       - f[ j    * procCoords.x_cells_num + i]) / hy
                ) / hy;
        }
    } else {
        for (i = 1; i < procCoords.x_cells_num -1; i++){
            delta_f[j * procCoords.x_cells_num + i] = 0;
        }
    }

    // ================================
    // Counting delta_f's corners
    // ================================    

    j = 0;
    i = 0;
    if (not procCoords.top and not procCoords.left) {
        delta_f[j * procCoords.x_cells_num + i] = (
                (f[j * procCoords.x_cells_num + i  ] - top_left_left                      ) / hx -
                (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
            ) / hx + (
                (f[ j    * procCoords.x_cells_num + i] - top_left_up                          ) / hy -
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
                (top_right_right                     - f[j * procCoords.x_cells_num + i  ]) / hx
            ) / hx + (
                (f[ j    * procCoords.x_cells_num + i] - top_right_up                         ) / hy -
                (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
            ) / hy;
    } else {
        delta_f[j * procCoords.x_cells_num + i] = 0;
    }

    j = procCoords.y_cells_num -1;
    i = 0;
    if (not procCoords.bottom and not procCoords.left){
        delta_f[j * procCoords.x_cells_num + i] = (
                (f[j * procCoords.x_cells_num + i  ] - bottom_left_left                   ) / hx -
                (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
            ) / hx + (
                (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                (bottom_left_down                      - f[ j    * procCoords.x_cells_num + i]) / hy
            ) / hy;
    } else {
        delta_f[j * procCoords.x_cells_num + i] = 0;
    }

    j = procCoords.y_cells_num -1;
    i = procCoords.x_cells_num -1;
    if (not procCoords.bottom and not procCoords.right){
        delta_f[j * procCoords.x_cells_num + i] = (
                (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                (bottom_right_right                  - f[j * procCoords.x_cells_num + i  ]) / hx
            ) / hx + (
                (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                (bottom_right_down                     - f[ j    * procCoords.x_cells_num + i]) / hy
            ) / hy;
    } else {
        delta_f[j * procCoords.x_cells_num + i] = 0;
    }

}

// ==================================================================================================================================================
//                                                                                                                                  DHP_PE_RA_FDM::fi
// ==================================================================================================================================================
bool DHP_PE_RA_FDM::stopCriteria (  const double* const f1, const double* const f2,
                                    const ProcParams& procParams, const ProcComputingCoords& procCoords){

    double norm = 0;
    for (uint i = 0; i < procCoords.x_cells_num * procCoords.y_cells_num; i++){
        norm = max(norm, abs(f1[i] - f2[i]));
    }

    if (gather_double_per_process == nullptr and procParams.rank == 0)
        gather_double_per_process = new double [procParams.size];

    int ret = MPI_Gather(
        &norm,                      // const void *sendbuf,
        1,                          // int sendcount,
        MPI_DOUBLE,                 // MPI_Datatype sendtype,
        gather_double_per_process,  // void *recvbuf,
        1,                          // int recvcount, (per process !)
        MPI_DOUBLE,                 // MPI_Datatype recvtype,
        0,                          // int root,
        procParams.comm             // MPI_Comm comm
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error gathering norm for checking stop criteria.");

    bool stop;
    if (procParams.rank == 0){
        norm = 0;
        for (uint i = 0; i < procParams.size; i++){
            norm = max(norm, gather_double_per_process[i]);
        }
        stop = norm < eps;

        cout << "stop= " << stop << " norm= " << norm << " eps= " << eps << endl;
    }

    ret = MPI_Bcast(
        &stop,          // void *buffer,
        1,              // int count,
        MPI_C_BOOL,     // MPI_Datatype datatype,
        0,              // int root, 
        procParams.comm // MPI_Comm comm
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error broadcasting value.");

    return stop;
}


// ==================================================================================================================================================
//                                                                                                                                DHP_PE_RA_FDM::Dump
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Dump_func(const string& fout_name, const ProcParams& procParams, const ProcComputingCoords& procCoords,
                         const double* const f, const string& func_label) const{

    if (debug){

        if (procParams.rank != 0) {
            MPI_Status status;
            char fiction;

            MPI_Recv(
                &fiction,                   // void *buf
                1,                          // int count
                MPI_CHAR,                   // MPI_Datatype datatype
                procParams.rank -1,         // int source
                MPI_MessageTypes::DumpSync, // int tag
                procParams.comm,            // MPI_Comm comm
                &status                     // MPI_Status *status
            );
        }

        fstream fout (fout_name, fstream::out | fstream::app);
        fout << "========== Dump begin ==========" << endl;

        fout << "procRank= " << procParams.rank << endl
            << "iterationsCounter= " << iterations_counter << endl << endl;

        if (f != nullptr){
            fout << endl << func_label << " (z x y)" << endl;// << std::fixed;

            for (uint j = 0; j < procCoords.y_cells_num; j++){
                for (uint i = 0; i < procCoords.x_cells_num; i++){
                    
                    fout << std::setprecision(3)
                        << setw (4) << X1 + (procCoords.x_cell_pos + i) * hx << " "
                        << setw (4) << Y1 + (procCoords.y_cell_pos + j) * hy << " ";

                    fout << std::setprecision(std::numeric_limits<double>::max_digits10) << f[j * procCoords.x_cells_num + i] << " ";

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
                MPI_MessageTypes::DumpSync, // int tag
                procParams.comm             // MPI_Comm comm
            );
        }

        if (procParams.rank == 0){
            string fiction;
            // cin >> fiction;
        }

        MPI_Barrier (procParams.comm);
    }
}


// ==================================================================================================================================================
//                                                                                                                                DHP_PE_RA_FDM::Dump
// ==================================================================================================================================================
void ProcComputingCoords::Dump(const string& fout_name) const{
    fstream fout (fout_name, fstream::out | fstream::app);

    fout << "x_proc_num= " << x_proc_num << " y_proc_num= " << y_proc_num << endl
        << "x_cells_num= " << x_cells_num << " x_cell_pos= " << x_cell_pos << endl
        << "y_cells_num= " << y_cells_num << " y_cell_pos= " << y_cell_pos << endl
        << "top= " << top << " bottom= " << bottom << " left= " << left << " right= " << right << endl;

    fout.close();
}
