#include <exception>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cmath>
#include <limits>
#include <iomanip>

#include <mpi.h>

#include <sys/time.h> // gettimeofday
#include <unistd.h> // sleep


#include "DHP_PE_RA_FDM.h"


using std::exception;

using std::cout;
using std::endl;

using std::fstream;
using std::stringstream;
using std::string;
using std::to_string;

using std::setprecision;
using std::numeric_limits;

// ==================================================================================================================================================
//                                                                                                                            DHP_PE_RA_FDM_superprac
// ==================================================================================================================================================
class DHP_PE_RA_FDM_superprac : public DHP_PE_RA_FDM {

        public:

    DHP_PE_RA_FDM_superprac(const uint grid_size_, const double eps_):
        DHP_PE_RA_FDM(0, 0, 3, 3, grid_size_, eps_, 1) {}

    void Print_p (const string& dout_name, const ProcParams& procParams, const uint x_proc_num, const uint y_proc_num) const;

        private:

    virtual double F (const double x, const double y) const;
    virtual double fi (const double x, const double y) const;

    // Exact solution
    // DSolveValue[{    Laplasian(u[x,y], {x, y}) = -(x*x + y*y)/((1+x*y)*(1+x*y))    ,    DirichletCondition[ u[x, y] = Piecewise[{{ln(1+x*y) , (y == 0 || y == 3) && ( 0 <= x && x <= 3) || (x == 0 || x == 3) && ( 0 <= y && y <= 3) }}], True];    }, u[x, y], {x, y} \[Element]    Rectangle[{0, 3}, {0, 3}]    ]

};

// ==================================================================================================================================================
// ==================================================================================================================================================
// Computing grid fragmentation between processes. I expect specific amount of processes
// 
void CoputeGridFragmentation (const ProcParams& procParams, uint& x_proc_num, uint& y_proc_num);

// ==================================================================================================================================================
//                                                                                                                                               MAIN
// ==================================================================================================================================================
int main (int argc, char** argv){

    if (argc != 4) {
        cout << "Wrong arguments. Example: ./program grid_size eps output.txt" << endl;
        return 1;
    }

    uint grid_size; stringstream(argv[1]) >> grid_size;
    double eps; stringstream(argv[2]) >> eps;
    string fout_name = string(argv[3]);

    // ================================
    MPI_Init(&argc,&argv); // Start MPI
    // ================================

    try {
        ProcParams procParams (MPI_COMM_WORLD);

        uint x_proc_num = 0;
        uint y_proc_num = 0;
        CoputeGridFragmentation (procParams, x_proc_num, y_proc_num);

        MPI_Comm currentComm;
        if (procParams.rank < x_proc_num * y_proc_num){
            MPI_Comm_split(MPI_COMM_WORLD, 1, procParams.rank, &currentComm);
        } else {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, procParams.rank, &currentComm);
        }

        if (currentComm != MPI_COMM_NULL){

            procParams = ProcParams(currentComm);

            DHP_PE_RA_FDM_superprac superPrac_2 (grid_size, eps);

            struct timeval tp;
            gettimeofday(&tp, NULL);
            long int start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

            superPrac_2.Compute(procParams, x_proc_num, y_proc_num);

            gettimeofday(&tp, NULL);
            long int finish_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

            if (procParams.rank == 0){
                cout << "output= " << fout_name << endl
                     << "x_proc_num= " << x_proc_num << endl
                     << "y_proc_num= " << y_proc_num << endl
                     << "miliseconds= " << finish_ms - start_ms << endl
                     << "iterationsCounter= " <<  superPrac_2.getIterationsCounter() << endl;
            }

            superPrac_2.Print_p(fout_name, procParams, x_proc_num, y_proc_num);

        }

        MPI_Comm_free (&currentComm);

    }
    catch (exception& e) {
        cout << e.what() << endl;
    }

    // ========================
    MPI_Finalize(); // Stop MPI
    // ========================

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
void DHP_PE_RA_FDM_superprac::Print_p (const string& dout_name, const ProcParams& procParams, const uint x_proc_num, const uint y_proc_num) const{

    ProcComputingCoords procCoords (procParams, grid_size, x_proc_num, y_proc_num);

    fstream fout(string("./") + dout_name + string("/output.txt.") + to_string(procParams.rank), fstream::out);
    // FILE* fout = fopen(dout_name.c_str(), "a");

    // This will block until file will get free
    // while (flock (fileno(fout), LOCK_EX) != 0){
    //     usleep(100000); // = 0.1 second // microseconds
    // }

    double* p = getSolutionPerProcess();

    // fprintf(fout, "# x y z\n");
    // fout << "# x y z" << endl;
    fout << "[" << endl;
    for (uint j = 0; j < procCoords.y_cells_num; j++){
        for (uint i = 0; i < procCoords.x_cells_num; i++){

            fout << "{"
                << "\"x\":" << std::setprecision(std::numeric_limits<double>::max_digits10) << X1 + (procCoords.x_cell_pos + i) * hx
                << ", \"y\":" << " " << std::setprecision(std::numeric_limits<double>::max_digits10) << Y1 + (procCoords.y_cell_pos + j) * hy
                << ", \"u\":" << " " << std::setprecision(std::numeric_limits<double>::max_digits10) << p[j * procCoords.x_cells_num + i];
            if (i == procCoords.x_cells_num -1 and j == procCoords.y_cells_num -1)
                fout << "}" << endl;
            else
                fout << "}," << endl;
            // fprintf (fout, "%f %f %f\n", X1 + (procCoords.x_cell_pos + i) * hx, Y1 + (procCoords.y_cell_pos + j) * hy, p[j * procCoords.x_cells_num + i]);
        }
        // fprintf (fout, "\n");
    }
    fout << "]" << endl;

    // free file lock
    // flock (fileno(fout), LOCK_UN);

    fout.close();
    // fclose(fout);
}


// ==================================================================================================================================================
//                                                                                                                            CoputeGridFragmentation
// ==================================================================================================================================================
void CoputeGridFragmentation (const ProcParams& procParams, uint& x_proc_num, uint& y_proc_num){

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
    }  else if (procParams.size >= 32) {
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
