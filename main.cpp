#include <exception>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cmath>
#include <limits>
#include <iomanip>


#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif


#include <sys/time.h> // gettimeofday


#include "DHP_PE_RA_FDM.h"


using std::exception;

using std::cout;
using std::endl;

using std::fstream;
using std::stringstream;
using std::string;

using std::setprecision;
using std::numeric_limits;

// ==================================================================================================================================================
//                                                                                                                            DHP_PE_RA_FDM_superprac
// ==================================================================================================================================================
class DHP_PE_RA_FDM_superprac : public DHP_PE_RA_FDM {

        public:

    DHP_PE_RA_FDM_superprac(const int grid_size_x, const int grid_size_y, const double eps_):
        DHP_PE_RA_FDM(0, 0, 3, 3, grid_size_x, grid_size_y, eps_, 1) {}

    void Print_p (const string& dout_name) const;

        private:

    double F (const double x, const double y) const;
    double fi (const double x, const double y) const;

};

// ==================================================================================================================================================
// ==================================================================================================================================================
// Computing grid fragmentation between processes. I expect specific amount of processes
// 
void ComputeGridFragmentation (const ProcParams& procParams, int& x_proc_num, int& y_proc_num);

// ==================================================================================================================================================
//                                                                                                                                               MAIN
// ==================================================================================================================================================
int main (int argc, char** argv){

    if (argc != 4) {
        cout << "Wrong arguments. Example: ./program grid_size eps output.txt" << endl;
        return 1;
    }

    int grid_size; stringstream(argv[1]) >> grid_size;
    double eps; stringstream(argv[2]) >> eps;
    string fout_name = string(argv[3]);

    MPI_Init(&argc,&argv);

    try {
        ProcParams procParams (MPI_COMM_WORLD);

        int x_proc_num = 0;
        int y_proc_num = 0;
        ComputeGridFragmentation (procParams, x_proc_num, y_proc_num);

        if (procParams.rank == 0){
            #ifdef _OPENMP
                cout << "OpenMP-version= " << _OPENMP << " Max-threads= " << omp_get_max_threads() << endl;
            #endif
        }

        DHP_PE_RA_FDM_superprac superPrac_2 (grid_size, grid_size, eps);

        struct timeval tp;
        gettimeofday(&tp, NULL);
        long int start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

        superPrac_2.Compute(procParams, x_proc_num, y_proc_num);

        gettimeofday(&tp, NULL);
        long int finish_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

        if (procParams.rank == 0){
            int ms = finish_ms - start_ms;
            cout << "out= " << fout_name << endl
                 << "x_p_N= " << x_proc_num << endl
                 << "y_p_N= " << y_proc_num << endl
                 << "ms= " << finish_ms - start_ms << " m:s= " << ms / 1000 / 60 << ":" << ms / 1000 % 60 <<  endl
                 << "it= " <<  superPrac_2.getIterationsCounter() << endl;
        }

        superPrac_2.Print_p(fout_name);

    }
    catch (exception& e) {
        cout << e.what() << endl;

        // MPI_Abort(MPI_COMM_WORLD, 2);
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}


// ==================================================================================================================================================
//                                                                                                                         DHP_PE_RA_FDM_superprac::F
// ==================================================================================================================================================
double DHP_PE_RA_FDM_superprac::F (const double x, const double y) const{
    double t = 1 + x*y;
    if (t == 0)
        throw DHP_PE_RA_FDM_Exception("Error computing 'F' function.");
    return (x*x + y*y)/(t*t);
}


// ==================================================================================================================================================
//                                                                                                                        DHP_PE_RA_FDM_superprac::fi
// ==================================================================================================================================================
double DHP_PE_RA_FDM_superprac::fi (const double x, const double y) const{
    double t = 1 + x*y;
    if (t <= 0)
        throw DHP_PE_RA_FDM_Exception("Error computing 'fi' function.");
    return std::log(1 + x*y);
}


// ==================================================================================================================================================
//                                                                                                                   DHP_PE_RA_FDM_superprac::Print_p
// ==================================================================================================================================================
void DHP_PE_RA_FDM_superprac::Print_p (const string& dout_name) const{

    ProcParams procParams = getProcParams();
    ProcComputingCoords procCoords = getProcCoords();

    stringstream ss;
    ss << "./" << dout_name << "/output.txt." << procParams.rank;
    fstream fout(ss.str().c_str(), fstream::out);

    double* p = getSolutionPerProcess();

    fout << "[" << endl;
    for (int j = 0; j < procCoords.y_cells_num; j++){
        for (int i = 0; i < procCoords.x_cells_num; i++){

            // std::numeric_limits<double>::max_digits10 - is a c++11 feature
            // std::numeric_limits<double>::max_digits10 == 17

            fout << "{"
                << "\"x\":" << std::setprecision(17) << X1 + (procCoords.x_cell_pos + i) * hx
                << ",\"y\":" << std::setprecision(17) << Y1 + (procCoords.y_cell_pos + j) * hy
                << ",\"u\":" << std::setprecision(17) << p[j * procCoords.x_cells_num + i];
            if (i == procCoords.x_cells_num -1 and j == procCoords.y_cells_num -1)
                fout << "}" << endl;
            else
                fout << "}," << endl;
        }
    }
    fout << "]" << endl;

    fout.close();
}


// ==================================================================================================================================================
//                                                                                                                           ComputeGridFragmentation
// ==================================================================================================================================================
void ComputeGridFragmentation (const ProcParams& procParams, int& x_proc_num, int& y_proc_num){

    if (procParams.size >= 1024) {
        x_proc_num = 32;
        y_proc_num = 32;
    } else if (procParams.size >= 512) {
        x_proc_num = 16;
        y_proc_num = 32;
    } else if (procParams.size >= 256) {
        x_proc_num = 16;
        y_proc_num = 16;
    } else if (procParams.size >= 128) {
        x_proc_num = 8;
        y_proc_num = 16;
    } else if (procParams.size >= 64) {
        x_proc_num = 8;
        y_proc_num = 8;
    } else if (procParams.size >= 32) {
        x_proc_num = 4;
        y_proc_num = 8;
    } else if (procParams.size >= 16) {
        x_proc_num = 4;
        y_proc_num = 4;
    } else if (procParams.size >= 8){
        x_proc_num = 2;
        y_proc_num = 4;
    } else if (procParams.size >= 4) {
        x_proc_num = 2;
        y_proc_num = 2;
    } else if (procParams.size >= 2) {
        x_proc_num = 1;
        y_proc_num = 2;
    } else if (procParams.size >= 1) {
        x_proc_num = 1;
        y_proc_num = 1;
    } else {
        throw DHP_PE_RA_FDM_Exception("Wrong process amount given for computations");
    }
}
