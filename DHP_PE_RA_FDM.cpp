#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include <cmath>
#include <limits>
#include <iomanip>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#include "DHP_PE_RA_FDM.h"


using std::cout;
using std::fstream;
using std::endl;
using std::setw;

using std::ceil;
using std::max;
using std::abs;

using std::swap;
using std::memset;

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
                                const double eps_, const int descent_step_iterations_):
X1(x1), Y1(y1), 
X2(x2), Y2(y2),

hx ((x2-x1)/grid_size_x_),
hy ((y2-y1)/grid_size_y_),

grid_size_x (grid_size_x_),
grid_size_y (grid_size_y_),
eps (eps_),
descent_step_iterations (descent_step_iterations_),
iterations_counter (0),

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
gather_double_per_process (NULL),

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
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::~DHP_PE_RA_FDM
// ==================================================================================================================================================
DHP_PE_RA_FDM::~DHP_PE_RA_FDM (){
    if (p != NULL){
        delete [] p; p = NULL;
    }
    if (p_prev != NULL){
        delete [] p_prev; p_prev = NULL;
    }
    if (send_message_lr != NULL){
    delete [] send_message_lr; send_message_lr = NULL;
    }
    if (send_message_rl != NULL){
        delete [] send_message_rl; send_message_rl = NULL;
    }
    if (send_message_td != NULL){
        delete [] send_message_td; send_message_td = NULL;
    }
    if (send_message_bu != NULL){
        delete [] send_message_bu; send_message_bu = NULL;
    }
    if (recv_message_lr != NULL){
        delete [] recv_message_lr; recv_message_lr = NULL;
    }
    if (recv_message_rl != NULL){
        delete [] recv_message_rl; recv_message_rl = NULL;
    }
    if (recv_message_td != NULL){
        delete [] recv_message_td; recv_message_td = NULL;
    }
    if (recv_message_bu != NULL){
        delete [] recv_message_bu; recv_message_bu = NULL;
    }
    if (recv_reqs_5_star != NULL){
        delete [] recv_reqs_5_star; recv_reqs_5_star = NULL;
    }
    if (send_reqs_5_star != NULL){
        delete [] send_reqs_5_star; send_reqs_5_star = NULL;
    }
    if (gather_double_per_process != NULL){
        delete [] gather_double_per_process; gather_double_per_process = NULL;
    }
    if (procParams.comm != MPI_COMM_WORLD){
        MPI_Comm_free(&procParams.comm);
    }
}


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::PrepareMPIComm
// ==================================================================================================================================================
MPI_Comm DHP_PE_RA_FDM::PrepareMPIComm(const ProcParams& procParams_in, const int x_proc_num, const int y_proc_num) const{

    if (procParams_in.size < x_proc_num * y_proc_num)
        throw DHP_PE_RA_FDM_Exception("Not enough processes for requested computations.");

    // cout << procParams_in.size << " " << grid_size_x << " " << grid_size_y << endl ;
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
        delete [] p; p = NULL;
    if (p_prev != NULL)
        delete [] p_prev; p_prev = NULL;

    p = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    p_prev = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* g = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* r = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_p = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_g = new double [procCoords.x_cells_num * procCoords.y_cells_num];
    double* delta_r = new double [procCoords.x_cells_num * procCoords.y_cells_num];

    double scalar_product_delta_g_and_g = 1;

    // Computing step 1
    Initialize_P_and_Pprev();
    Dump_func(debug_fname, p_prev, "p_prev");


    iterations_counter = 0;
    while (true) {
        if (debug and procParams.rank == 0)
            cout << endl << "iterations_counter = " << iterations_counter << endl;

        // Computing step 2
        Counting_5_star (p_prev, delta_p);
        Dump_func(debug_fname, delta_p, "delta_p");

        // Computing step 3
        Compute_r (delta_p, r);
        Dump_func(debug_fname, r, "r");

        double scalar_product_delta_r_and_g = 1;
        if (iterations_counter >= descent_step_iterations){
            // Computing step 4
            Counting_5_star (r, delta_r);
            Dump_func(debug_fname, delta_r, "delta_r");

            // Computing step 5
            scalar_product_delta_r_and_g = ComputingScalarProduct(delta_r, g);
            if (debug and procParams.rank == 0)
                cout << "scalar_product_delta_r_and_g= " << scalar_product_delta_r_and_g << endl;
        }

        // Computing step 6
        double alpha = 0;
        if (iterations_counter >= descent_step_iterations){
            if (procParams.rank == 0){
                alpha = scalar_product_delta_r_and_g / scalar_product_delta_g_and_g;
            }
            alpha = BroadcastParameter(alpha);
            if (debug and procParams.rank == 0)
                cout << "alpha= " << alpha << endl;
        }

        // Computing step 7
        Compute_g (g, r, alpha);
        Dump_func(debug_fname, g, "g");

        // Computing step 8
        Counting_5_star (g, delta_g);
        Dump_func(debug_fname, delta_g, "delta_g");

        // Computing step 9
        double scalar_product_r_and_g = ComputingScalarProduct(r, g);
        if (debug and procParams.rank == 0)
            cout << "scalar_product_r_and_g= " << scalar_product_r_and_g << endl;

        // Computing step 10
        scalar_product_delta_g_and_g = ComputingScalarProduct(delta_g, g);
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
        tau = BroadcastParameter(tau);
        if (debug and procParams.rank == 0)
            cout << "tau= " << tau << endl;

        if (tau == -1){
            throw DHP_PE_RA_FDM_Exception( "Error. there is a divizion by zero in computations. Zero value is scalar product of delta_g and g.\n"
                "Probably there can be next problems:\n"
                "\t1) You have to increase amount of descent step iterations (default: 1);\n"
                "\t2) Algorithm has problems with fraction(fixed point part) of type 'double'.\n"
                "\t3) You specified too small matrix (internal region does not contain internal points, the only points are in boundary region).\n"
                "Iteration process terminated.");

            // 'p' - is side effect of computation and within error,
            // last correct computation will be taken from p_prev
            swap(p, p_prev);
            break;
        }

        // Computing step 12
        Compute_p (tau, g);
        Dump_func(debug_fname, p, "p");

        if (stopCriteria (p, p_prev))
            break;

        swap(p, p_prev);
        iterations_counter++;
    }

    delete [] g; g = NULL;
    delete [] r; r = NULL;
    delete [] delta_p; delta_p = NULL;
    delete [] delta_g; delta_g = NULL;
    delete [] delta_r; delta_r = NULL;
}

// ==================================================================================================================================================
//                                                                                                                  DHP_PE_RA_FDM::BroadcastParameter
// ==================================================================================================================================================
double DHP_PE_RA_FDM::BroadcastParameter (double param){

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
double DHP_PE_RA_FDM::ComputingScalarProduct(const double* const f, const double* const delta_f){

    double scalar_product = 0;
    #pragma omp parallel
    {
        // #pragma omp for schedule (static) reduction(+:scalar_product) collapse (2)
        #pragma omp for schedule (static) reduction(+:scalar_product)
        for (int j = 0; j < procCoords.y_cells_num; j++){
            for (int i = 0; i < procCoords.x_cells_num; i++){
                scalar_product += hx * hy * f[j * procCoords.x_cells_num + i] * delta_f[j * procCoords.x_cells_num + i];
            }
        }
    }

    if (gather_double_per_process == NULL and procParams.rank == 0)
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
        #pragma omp parallel
        {
            #pragma omp for schedule (static) reduction(+:scalar_product)
            for (int i = 0; i < procParams.size; i++){
                scalar_product += gather_double_per_process[i];
            }
        }
        return scalar_product;
    } else {
        return 0;
    }
}


// ==================================================================================================================================================
//                                                                                                                     DHP_PE_RA_FDM::Counting_5_star
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Counting_5_star (const double* const f, double* const delta_f){

    int i = 0;
    int j = 0;
    int ret = MPI_SUCCESS;

    #pragma omp parallel
    {
        // #pragma omp for schedule (static) collapse (2)
        #pragma omp for schedule (static) private(i)
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
    }

    // ==========================================
    // memory allocation
    // ==========================================

    if (send_message_lr == NULL)
        send_message_lr = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (send_message_rl == NULL)
        send_message_rl = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (send_message_td == NULL)
        send_message_td = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (send_message_bu == NULL)
        send_message_bu = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (recv_message_lr == NULL)
        recv_message_lr = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (recv_message_rl == NULL)
        recv_message_rl = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (recv_message_td == NULL)
        recv_message_td = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];
    if (recv_message_bu == NULL)
        recv_message_bu = new double [max(procCoords.x_cells_num, procCoords.y_cells_num)];

    // ==========================================
    // initialize send buffers
    // ==========================================

    #pragma omp sections private(i, j)
    {
        // left -> right
        #pragma omp section
        for (j = 0; j < procCoords.y_cells_num; j++){
            send_message_lr[j] = f[ (j+1) * procCoords.x_cells_num -1];
        }
        // right -> left
        #pragma omp section
        for (j = 0; j < procCoords.y_cells_num; j++){
            send_message_rl[j] = f[ j * procCoords.x_cells_num + 0];
        }
        // top -> down
        #pragma omp section
        for (i = 0; i < procCoords.x_cells_num; i++){
            send_message_td[i] = f[ (procCoords.y_cells_num -1) * procCoords.x_cells_num + i];
        }
        // bottom -> up
        #pragma omp section
        for (i = 0; i < procCoords.x_cells_num; i++){
            send_message_bu[i] = f[i];
        }
    }


    int send_amount = 0;
    int recv_amount = 0;

    #pragma omp sections
    {
        // ==========================================
        // send messages
        // ==========================================
        #pragma omp section
        {

            // left -> right
            if (not procCoords.right){

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
        } // #pragma omp section


        // ==========================================
        // receive messages
        // ==========================================
        #pragma omp section
        {

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
        } // #pragma omp section

    } // #pragma omp sections

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
        i = 0;
        if (not procCoords.left) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]               ) / hx -
                            (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                            (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // right -> left
        i = procCoords.x_cells_num -1;
        if (not procCoords.right) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                            (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i  ]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                            (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // top -> down
        j = 0;
        if (not procCoords.top) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                            (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - recv_message_td[i]               ) / hy -
                            (f[(j+1) * procCoords.x_cells_num + i] - f[j * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // bottom -> up
        j = procCoords.y_cells_num -1;
        if (not procCoords.bottom) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                            (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                            (recv_message_bu[i]                    - f[ j    * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
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
            }

            #pragma omp section
            {
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
            }

            #pragma omp section
            {
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
            }

            #pragma omp section
            {
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
            }
        } // #pragma omp sections

    } else if (procCoords.x_cells_num > 1 and procCoords.y_cells_num == 1){
        // Counting regions n x 1, where n > 1

        // top -> down
        // bottom -> up
        j = 0;
        if (not procCoords.top and not procCoords.bottom) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - f[j * procCoords.x_cells_num + i-1]) / hx -
                            (f[j * procCoords.x_cells_num + i+1] - f[j * procCoords.x_cells_num + i  ]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - recv_message_td[i]               ) / hy -
                            (recv_message_bu[i]                    - f[j * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (i = 1; i < procCoords.x_cells_num -1; i++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
                j = 0;
                i = 0;
                if (not procCoords.top and not procCoords.bottom) {
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
            }

            #pragma omp section
            {
                j = 0;
                i = procCoords.x_cells_num -1;
                if (not procCoords.top and not procCoords.bottom){
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
            }
        } // #pragma omp sections

    } else if (procCoords.x_cells_num == 1 and procCoords.y_cells_num > 1){
        // Counting regions 1 x m, where m > 1

        // left -> right
        // right -> left
        i = 0;
        if (not procCoords.left and not procCoords.right) {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = (
                            (f[j * procCoords.x_cells_num + i  ] - recv_message_lr[j]               ) / hx -
                            (recv_message_rl[j]                  - f[j * procCoords.x_cells_num + i]) / hx
                        ) / hx + (
                            (f[ j    * procCoords.x_cells_num + i] - f[(j-1) * procCoords.x_cells_num + i]) / hy -
                            (f[(j+1) * procCoords.x_cells_num + i] - f[ j    * procCoords.x_cells_num + i]) / hy
                        ) / hy;
                }
            }
        } else {
            #pragma omp parallel
            {
                #pragma omp for schedule (static)
                for (j = 1; j < procCoords.y_cells_num -1; j++){
                    delta_f[j * procCoords.x_cells_num + i] = 0;
                }
            }
        }

        // ==========================================
        // Counting delta_f's corners
        // ==========================================

        #pragma omp sections private(i, j)
        {
            #pragma omp section
            {
                j = 0;
                i = 0;
                if (not procCoords.left and not procCoords.right) {
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
            }

            #pragma omp section
            {
                j = procCoords.y_cells_num -1;
                i = 0;
                if (not procCoords.left and not procCoords.right){
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
            }
        } // #pragma omp sections

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

}

// ==================================================================================================================================================
//                                                                                                                                  DHP_PE_RA_FDM::fi
// ==================================================================================================================================================
bool DHP_PE_RA_FDM::stopCriteria(const double* const f1, const double* const f2){

    double norm = 0;
    // #pragma omp parallel reduction(max:norm)
    double priv_norm = 0;
    #pragma omp parallel firstprivate (priv_norm)
    {
        #pragma omp for schedule (static)
        for (int i = 0; i < procCoords.x_cells_num * procCoords.y_cells_num; i++){
            priv_norm = max(priv_norm, abs(f1[i] - f2[i]));
        }

        #pragma omp critical
        {
            norm = max(priv_norm, norm);
        }
    }

    if (gather_double_per_process == NULL and procParams.rank == 0)
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

    int stop;
    if (procParams.rank == 0){
        norm = 0;
        // #pragma omp parallel reduction(max:norm)
        double priv_norm = 0;
        #pragma omp parallel firstprivate (priv_norm)
        {
            #pragma omp for schedule (static)
            for (int i = 0; i < procParams.size; i++){
                priv_norm = max(priv_norm, gather_double_per_process[i]);
            }

            #pragma omp critical
            {
                norm = max(priv_norm, norm);
            }
        }

        stop = static_cast<int>(norm < eps);

        if (debug)
            cout << "stop= " << stop << " norm= " << norm << " eps= " << eps << endl;
    }

    ret = MPI_Bcast(
        &stop,          // void *buffer,
        1,              // int count,
        MPI_INT,        // MPI_Datatype datatype,
        0,              // int root, 
        procParams.comm // MPI_Comm comm
    );

    if (ret != MPI_SUCCESS) throw DHP_PE_RA_FDM_Exception("Error broadcasting value.");

    return stop;
}


// ==================================================================================================================================================
//                                                                                                              DHP_PE_RA_FDM::Initialize_P_and_Pprev
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Initialize_P_and_Pprev (){
    #pragma omp parallel
    {
        // boundary region
        if (procCoords.top){
            #pragma omp for schedule (guided)
            for (int i = 0; i < procCoords.x_cells_num; i++){
                p_prev[0 * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + 0) * hy);
                p[0 * procCoords.x_cells_num + i] = fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + 0) * hy);
            }
        }
        if (procCoords.bottom){
            #pragma omp for schedule (guided)
            for (int i = 0; i < procCoords.x_cells_num; i++){
                p_prev[(procCoords.y_cells_num -1) * procCoords.x_cells_num + i] =
                    fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + (procCoords.y_cells_num -1)) * hy);
                p[(procCoords.y_cells_num -1) * procCoords.x_cells_num + i] =
                    fi(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + (procCoords.y_cells_num -1)) * hy);
            }
        }
        if (procCoords.left){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                p_prev[j * procCoords.x_cells_num + 0] = fi(X1 + (procCoords.x_cell_pos + 0) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
                p[j * procCoords.x_cells_num + 0] = fi(X1 + (procCoords.x_cell_pos + 0) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
            }
        }
        if (procCoords.right){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                p_prev[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] =
                    fi(X1 + (procCoords.x_cell_pos + (procCoords.x_cells_num -1)) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
                p[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] =
                    fi(X1 + (procCoords.x_cell_pos + (procCoords.x_cells_num -1)) * hx, Y1 + (procCoords.y_cell_pos + j) * hy);
            }
        }

        // internal region
        #pragma omp for schedule (guided)
        for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
            memset(&(p_prev[j * procCoords.x_cells_num + static_cast<int>(procCoords.left)]), 0,
                procCoords.x_cells_num - static_cast<int>(procCoords.right) - static_cast<int>(procCoords.left));

            // for (int i = static_cast<int>(procCoords.left); i < procCoords.x_cells_num - static_cast<int>(procCoords.right); i++){
            //     p_prev[j * procCoords.x_cells_num + i] = 0;
            // }
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_r
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute_r (const double* const delta_p, double* const r) const{
    #pragma omp parallel
    {
        // boundary region
        if (procCoords.top){

            memset(&(r[(procCoords.y_cells_num -1) * procCoords.x_cells_num + 0]), 0,
                procCoords.x_cells_num);
        }
        if (procCoords.bottom){

            memset(&(r[(procCoords.y_cells_num -1) * procCoords.x_cells_num + 0]), 0,
                procCoords.x_cells_num);
        }
        if (procCoords.left){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                r[j * procCoords.x_cells_num + 0] = 0;
            }
        }
        if (procCoords.right){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                r[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] = 0;
            }
        }

        // internal region
        // #pragma omp for schedule (guided) collapse (2)
        #pragma omp for schedule (guided)
        for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
            for (int i = static_cast<int>(procCoords.left); i < procCoords.x_cells_num - static_cast<int>(procCoords.right); i++){
                r[j * procCoords.x_cells_num + i] =
                    delta_p[j * procCoords.x_cells_num + i] -
                    F(X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy)
                    ;
            }
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_g
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute_g (double* const g, const double* const r, const double alpha) const{
    #pragma omp parallel
    {
        // boundary region
        if (procCoords.top){

            memset(&(g[(procCoords.y_cells_num -1) * procCoords.x_cells_num + 0]), 0,
                procCoords.x_cells_num);
        }
        if (procCoords.bottom){

            memset(&(g[(procCoords.y_cells_num -1) * procCoords.x_cells_num + 0]), 0,
                procCoords.x_cells_num);
        }
        if (procCoords.left){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                g[j * procCoords.x_cells_num + 0] = 0;
            }
        }
        if (procCoords.right){
            #pragma omp for schedule (guided)
            for (int j = 0; j < procCoords.y_cells_num; j++){
                g[j * procCoords.x_cells_num + (procCoords.x_cells_num -1)] = 0;
            }
        }

        // internal region
        if (iterations_counter >= descent_step_iterations){
            // #pragma omp for schedule (guided) collapse (2)
            #pragma omp for schedule (guided)
            for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
                for (int i = static_cast<int>(procCoords.left); i < procCoords.x_cells_num - static_cast<int>(procCoords.right); i++){
                    g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i] - alpha * g[j * procCoords.x_cells_num + i];
                }
            }
        } else {
            // #pragma omp for schedule (guided) collapse (2)
            #pragma omp for schedule (guided)
            for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
                for (int i = static_cast<int>(procCoords.left); i < procCoords.x_cells_num - static_cast<int>(procCoords.right); i++){
                    g[j * procCoords.x_cells_num + i] = r[j * procCoords.x_cells_num + i];
                }
            }
        }   
    }
}


// ==================================================================================================================================================
//                                                                                                                           DHP_PE_RA_FDM::Compute_p
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Compute_p (const double tau, const double* const g) {
    #pragma omp parallel
    {
        // #pragma omp for schedule (guided) collapse (2)
        #pragma omp for schedule (guided)
        for (int j = static_cast<int>(procCoords.top); j < procCoords.y_cells_num - static_cast<int>(procCoords.bottom); j++){
            for (int i = static_cast<int>(procCoords.left); i < procCoords.x_cells_num - static_cast<int>(procCoords.right); i++){
                p[j * procCoords.x_cells_num + i] = p_prev[j * procCoords.x_cells_num + i] - tau * g[j * procCoords.x_cells_num + i];
            }
        }
    }
}


// ==================================================================================================================================================
//                                                                                                                                DHP_PE_RA_FDM::Dump
// ==================================================================================================================================================
void DHP_PE_RA_FDM::Dump_func(const string& fout_name, const double* const f, const string& func_label) const{

    if (debug){

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
            fout << endl << func_label << " (z x y)" << endl;// << std::fixed;

            for (int j = 0; j < procCoords.y_cells_num; j++){
                for (int i = 0; i < procCoords.x_cells_num; i++){
                    
                    fout << std::setprecision(3)
                        << setw (4) << X1 + (procCoords.x_cell_pos + i) * hx << " "
                        << setw (4) << Y1 + (procCoords.y_cell_pos + j) * hy << " ";

                    // std::numeric_limits<double>::max_digits10 - is a c++11 feature
                    // std::numeric_limits<double>::max_digits10 == 17
                    fout << std::setprecision(17) << f[j * procCoords.x_cells_num + i] << " ";

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
