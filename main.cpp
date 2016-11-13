#include <exception>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cmath>
#include <limits>
#include <iomanip>


#include <mpi.h>
#include <cuda_runtime.h>


#include <sys/time.h> // gettimeofday
#include <unistd.h> // sleep


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

    DHP_PE_RA_FDM_superprac(const int grid_size_x, const int grid_size_y, const double eps_, const int cudaDevNum):
        DHP_PE_RA_FDM(0, 0, 3, 3, grid_size_x, grid_size_y, eps_, cudaDevNum, 1) {}

    void Print_p (const string& dout_name);

};

// ==================================================================================================================================================
// ==================================================================================================================================================

// Computing grid fragmentation between processes. I expect specific amount of processes
// 
void ComputeGridFragmentation (const ProcParams& procParams, int& x_proc_num, int& y_proc_num);

void GetVideoCardProperties (const string& fout_name);

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

        if (procParams.rank == 0)
            GetVideoCardProperties("Tesla-X2070-TTX.txt");

        int x_proc_num = 0;
        int y_proc_num = 0;
        ComputeGridFragmentation (procParams, x_proc_num, y_proc_num);

        // I expect (procParams.rank % 2) to return different values for processes on one node
        // This program must be started only assumption, that amount of videocards equals to processes on each node
        DHP_PE_RA_FDM_superprac superPrac_2 (grid_size, grid_size, eps, procParams.rank % 2);

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
//                                                                                                                   DHP_PE_RA_FDM_superprac::Print_p
// ==================================================================================================================================================
void DHP_PE_RA_FDM_superprac::Print_p (const string& dout_name){

    ProcParams procParams = getProcParams();
    ProcComputingCoords procCoords = getProcCoords();

    stringstream ss;
    ss << "./" << dout_name << "/output.txt." << procParams.rank;
    fstream fout(ss.str().c_str(), fstream::out);

    double* p = getSolutionPerProcess();

    if (local_f == NULL)
        SAFE_CUDA(cudaHostAlloc(&local_f, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*local_f), cudaHostAllocMapped));

    SAFE_CUDA(cudaMemcpy(local_f, p, procCoords.x_cells_num * procCoords.y_cells_num * sizeof(*p), cudaMemcpyDeviceToHost));

    fout << "[" << endl;
    for (int j = 0; j < procCoords.y_cells_num; j++){
        for (int i = 0; i < procCoords.x_cells_num; i++){

            // std::numeric_limits<double>::max_digits10 - is a c++11 feature
            // std::numeric_limits<double>::max_digits10 == 17

            fout << "{"
                << "\"x\":" << std::setprecision(17) << X1 + (procCoords.x_cell_pos + i) * hx
                << ",\"y\":" << std::setprecision(17) << Y1 + (procCoords.y_cell_pos + j) * hy
                << ",\"u\":" << std::setprecision(17) << local_f[j * procCoords.x_cells_num + i];
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
//                                                                                                                             GetVideoCardProperties
// ==================================================================================================================================================
void GetVideoCardProperties (const string& fout_name){
    
    fstream fout (fout_name.c_str(), fstream::out);

    int deviceNum = 0;
    SAFE_CUDA(cudaGetDeviceCount(&deviceNum));

    fout << "deviceNum= " << deviceNum << endl;

    if (deviceNum > 0){
        cudaDeviceProp devProp;
        SAFE_CUDA(cudaGetDeviceProperties(&devProp, 0));

        fout
            << "videocard-name= " << devProp.name << endl
            << "cudaVersion= " << devProp.major << "." << devProp.minor << endl
            << endl
            << "multiProcessorCount= " << devProp.multiProcessorCount << endl
            << "warpSize= " << devProp.warpSize << endl
            << "maxThreadsPerBlock= " << devProp.maxThreadsPerBlock << endl
            << "maxGridSize= " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2] << endl
            << "maxThreadsDim= " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2] << endl
            << endl
            << "totalGlobalMem= " << devProp.totalGlobalMem << endl
            << "totalConstMem= " << devProp.totalConstMem << endl
            << "sharedMemPerBlock= " << devProp.sharedMemPerBlock << endl
            << "regsPerBlock= " << devProp.regsPerBlock << endl
            << endl
            << "canMapHostMemory= " << devProp.canMapHostMemory << endl
            << "unifiedAddressing= " << devProp.unifiedAddressing << endl
            << "computeMode= " << devProp.computeMode << endl
            << "asyncEngineCount= " << devProp.asyncEngineCount << endl
            << "concurrentKernels= " << devProp.concurrentKernels << endl
            << "deviceOverlap= " << devProp.deviceOverlap << endl
            ;
    }

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
