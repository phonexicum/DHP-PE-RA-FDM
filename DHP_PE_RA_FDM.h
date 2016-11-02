#include <exception>
#include <string>

#include <mpi.h>

using std::exception;
using std::string;

// ==================================================================================================================================================
//                                                                                                                            DHP_PE_RA_FDM_Exception
// ==================================================================================================================================================
class DHP_PE_RA_FDM_Exception : public exception {

        public:

    DHP_PE_RA_FDM_Exception(const string& msg_): msg (msg_) {}
    virtual ~DHP_PE_RA_FDM_Exception() throw() {}

    virtual const char* what() const noexcept{
        return msg.c_str();
    }

        private:

    const string msg;
};

// ==================================================================================================================================================
//                                                                                                                                         ProcParams
// ==================================================================================================================================================
struct ProcParams {

    uint rank;
    uint size;
    MPI_Comm comm;

        public:
    
    ProcParams(MPI_Comm comm_){
        comm = comm_;
        int s, r;
        MPI_Comm_rank (comm, &r); // get current process id
        MPI_Comm_size (comm, &s); // get number of processes
        if (r < 0 || s < 0) throw DHP_PE_RA_FDM_Exception ("I did not know that MPI can return negative size or rank.");
        rank = static_cast<uint>(r);
        size = static_cast<uint>(s);
    }
};

// ==================================================================================================================================================
//                                                                                                                                ProcComputingCoords
// ==================================================================================================================================================
// Structure for storing coords of computing area of corrent process
// 
struct ProcComputingCoords {

    uint x_proc_num;
    uint y_proc_num;

    uint x_cells_num;
    uint x_cell_pos;
    uint y_cells_num;
    uint y_cell_pos;

    // Indicates if the process touches the border
    bool top;
    bool bottom;
    bool left;
    bool right;

    ProcComputingCoords (const ProcParams& procParams, const uint grid_size_, const uint x_proc_num_, const uint y_proc_num_);

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
//      stopping criteria: function 'stopCriteria'
//
// Both: boundary conditions and right side of Laplace operator, must be defined in successor class.
// 
// In case various errors 'DHP_PE_RA_FDM_Exception' can be thrown
// 
// Dirichlet-Problem-Poisson's-Equation-Rectangular-Area-Finite-Difference-Method
class DHP_PE_RA_FDM {

        public:
    
    DHP_PE_RA_FDM(  const double x1, const double y1, const double x2, const double y2, const uint grid_size_, const double eps_,
                    const uint descent_step_iterations_ = 1);
    ~DHP_PE_RA_FDM();

    // Main function for solving differential equation
    // 
    //  after-effect: after function finished, each processor will have its own part of solution for target function
    //  solution can be found at 'double* p;'
    //  
    void Compute (const ProcParams& procParams, const uint x_proc_num, const uint y_proc_num);

    double* getSolutionPerProcess () const { return p; }
    uint getIterationsCounter () const { return iterations_counter; }

    void Dump_func(const string& fout_name, const ProcParams& procParams, const ProcComputingCoords& procCoords,
        const double* const f = nullptr, const string& func_label = string("")) const;

    const double X1;
    const double Y1;
    const double X2;
    const double Y2;

    const double hx;
    const double hy;
    
    const uint grid_size;
    const double eps;


        protected:

    // right side of Laplace operator
    virtual double F (const double x, const double y) const = 0;
    // boundary conditions
    virtual double fi (const double x, const double y) const = 0;
    // stopping criteria (must return true/false value for each process)
    virtual bool stopCriteria (const double* const f1, const double* const f2, const ProcParams& procParams, const ProcComputingCoords& procCoords);


        private:

    // This function computes five-point difference equation for Laplace operator approximation for function f and stores result into delta_f
    void Counting_5_star (const double* const f, double* const delta_f, const ProcParams& procParams, const ProcComputingCoords& procCoords);

    // This function computes scalar product of two functions. Scalar product ignores function values on the boundaries
    // 
    //  return value: scalar_product of two functions, if the process rank equals to 0, returns 0 otherwise
    //  
    double ComputingScalarProduct (const double* const f, const double* const delta_f, const ProcParams& procParams, const ProcComputingCoords& procCoords);

    // 
    // return value: broadcasted param gotten from process with rank == 0
    // 
    double BroadcastParameter (double param, const ProcParams& procParams);


    uint descent_step_iterations;
    uint iterations_counter;

    // p is a double array
    // p(i=x, j=y) = p [y * row_len + x]
    double* p;
    double* p_prev;

    // in spite of the fact that this variables is temporal and used only in Counting_5_star and ComputingScalarProduct functions,
    // they are too oftenly used, to be allocated on each iteration of computing process
    double* send_message;
    double* recv_message;
    double* gather_double_per_process;

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
