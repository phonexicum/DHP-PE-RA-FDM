#pragma once

#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include <mpi.h>
#include <cuda_runtime.h>

using std::exception;
using std::string;
using std::stringstream;
using std::pair;

#define SAFE_CUDA(err)\
    if (err != cudaSuccess){ \
        stringstream sstr; sstr << __FILE__ << ":" << __LINE__ << " cuda-error= " << cudaGetErrorString(err); \
        throw DHP_PE_RA_FDM_Exception(sstr.str().c_str()); \
    }

// ==================================================================================================================================================
//                                                                                                                            DHP_PE_RA_FDM_Exception
// ==================================================================================================================================================
class DHP_PE_RA_FDM_Exception : public exception {

        public:

    DHP_PE_RA_FDM_Exception(const string& msg_): msg (msg_) {}
    virtual ~DHP_PE_RA_FDM_Exception() throw() {}

    virtual const char* what() const throw() {
        return msg.c_str();
    }

        private:

    const string msg;
};

// ==================================================================================================================================================
//                                                                                                                                         ProcParams
// ==================================================================================================================================================
struct ProcParams {

    int rank;
    int size;
    MPI_Comm comm;

        public:
    
    ProcParams(MPI_Comm comm_ = MPI_COMM_WORLD){
        comm = comm_;
        MPI_Comm_rank (comm, &rank); // get current process id
        MPI_Comm_size (comm, &size); // get number of processes
    }
};

// ==================================================================================================================================================
//                                                                                                                                ProcComputingCoords
// ==================================================================================================================================================
// Structure for storing coords of computing area of corrent process
// 
struct ProcComputingCoords {

    int x_proc_num;
    int y_proc_num;

    int x_cells_num;
    int x_cell_pos;
    int y_cells_num;
    int y_cell_pos;

    // Indicates if the process touches the border
    bool top;
    bool bottom;
    bool left;
    bool right;

    ProcComputingCoords ();
    ProcComputingCoords (const ProcParams& procParams, const int grid_size_x, const int grid_size_y, const int x_proc_num_, const int y_proc_num_);

    void Dump(const string& fout_name) const;
};

// ==================================================================================================================================================
//                                                                                                                                      DHP_PE_RA_FDM
// ==================================================================================================================================================
// This class solves the Dirichlet problem for Poisson's equation in rectangular area using "steep descent iterations" for first several iterations
// and "conjugate gragient iterations" afterwards.
// 
//      method uses:
//          five-point difference equation for Laplace operator approximation
//          grid fragmentation are regular
//          MPI technology for counting under supercomputers
//          scalar product: (a, b) = \sum_{i=1}^{i=n-1} ( \sum_{j=1}^{j=m-1} ( h'i * h'j * a(i, j) * b(i, j) ))
// 
//      boundary conditions: function 'fi'
//      right side of Laplace operator: function 'F'
//      stopping criteria: function 'StopCriteria'
//
// Both: boundary conditions and right side of Laplace operator, must be defined in successor class.
// 
// In case various errors 'DHP_PE_RA_FDM_Exception' can be thrown (any process can throw an exception, however there is some warranties, that
//  exceptions relative to algorithmical problems (not MPI errors) will be thrown by each process)
// 
// Dirichlet-Problem-Poisson's-Equation-Rectangular-Area-Finite-Difference-Method
// 
// Я храню массивы внутри GPU
// 
// 
#define fi(x, y) (x*x + y*y)/((1 + x*y)*(1 + x*y))
#define F(x, y) logf(1 + x*y)
// 
class DHP_PE_RA_FDM {

        public:
    
    DHP_PE_RA_FDM(  const double x1, const double y1, const double x2, const double y2, const int grid_size_x_, const int grid_size_y_,
                    const double eps_, const int cudaDeviceNum, const int descent_step_iterations_ = 1);
    virtual ~DHP_PE_RA_FDM();

    // Main function for solving differential equation
    // 
    //  after-effect: after function finished, each processor will have its own part of solution for target function
    //  solution can be found at 'double* p;'
    //  
    void Compute (const ProcParams& procParams_in, const int x_proc_num, const int y_proc_num);

    double* getSolutionPerProcess () const { return p; }
    int getIterationsCounter () const { return iterations_counter; }
    ProcParams getProcParams () const { return procParams; }
    ProcComputingCoords getProcCoords () const { return procCoords; }

    void Dump_func(const string& fout_name, const double* const f = NULL, const string& func_label = string("")) const;

    const double X1;
    const double Y1;
    const double X2;
    const double Y2;

    const double hx;
    const double hy;
    
    const int grid_size_x;
    const int grid_size_y;
    const double eps;


        protected:

    virtual bool StopCriteria (const double* const f1, const double* const f2);

    MPI_Comm PrepareMPIComm (const ProcParams& procParams_in, const int x_proc_num, const int y_proc_num) const;


        private:

    cudaDeviceProp devProp;
    pair<dim3, dim3> GridDistribute (const int demandedThreadNum);
    void cudaAllStreamsSynchronize (const int begin = 0, const int end = (cudaStreams_num -1));


    // This function computes five-point difference equation for Laplace operator approximation for function f and stores result into delta_f
    void Counting_5_star (const double* const f, double* const delta_f);
    void cuda_Counting_5_star (const double* const f, double* const delta_f);

    // This function computes scalar product of two functions. Scalar product ignores function values on the boundaries
    // 
    //  return value: global scalar_product of two functions for each of the processes
    //  
    double ComputingScalarProduct (const double* const f, const double* const delta_f);

    void Initialize_P_and_Pprev ();
    void cuda_Initialize_P_and_Pprev ();

    void cuda_Initialize_F_border_with_zero (double* const f);
    void Compute_r (const double* const delta_p, double* const r) const;
    void cuda_Compute_r (double* const r, const double* const delta_p) const;
    void Compute_g (double* const g, const double* const r, const double alpha) const;
    void Compute_p (const double tau, const double* const g);


    ProcParams procParams;
    ProcComputingCoords procCoords;
    int descent_step_iterations;
    int iterations_counter;

    static const int cudaStreams_num = 16;
    cudaStream_t cudaStreams[cudaStreams_num];
    
    // p is a double array allocated ON DEVICE
    // p(i=x, j=y) = p [y * row_len + x]
    double* p;
    double* p_prev;

    // in spite of the fact that this variables is temporal and used only in Counting_5_star and ComputingScalarProduct functions,
    // they are too oftenly used, to be allocated on each iteration of computing process
    double* send_message_lr;
    double* send_message_rl;
    double* send_message_td;
    double* send_message_bu;
    double* recv_message_lr;
    double* recv_message_rl;
    double* recv_message_td;
    double* recv_message_bu;
    MPI_Request* recv_reqs_5_star;
    MPI_Request* send_reqs_5_star;

    enum MPI_MessageTypes {
        StarLeftRight,
        StarRightLeft,
        StarTopDown,
        StarBottomUp,
        DumpSync
    };

    static const bool debug = true;
    string debug_fname;
};
