#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                                      DHP_PE_RA_FDM::GridDistribute
// ==================================================================================================================================================
pair<dim3, dim3> DHP_PE_RA_FDM::GridDistribute (const int demandedThreadNum) const{

    dim3 blockDim = dim3(min(min(demandedThreadNum, devProp.maxThreadsPerBlock), devProp.maxThreadsDim[0]));

    int demandedBlockNum = (demandedThreadNum -1) / blockDim.x +1;
    if (demandedBlockNum >= devProp.maxGridSize[0])
        throw DHP_PE_RA_FDM_Exception("Too many number of threads for device demanded.");

    dim3 gridDim = dim3(demandedBlockNum);
    
    return make_pair(gridDim, blockDim);
}


// ==================================================================================================================================================
//                                                                                                           DHP_PE_RA_FDM::cudaAllStreamsSynchronize
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cudaAllStreamsSynchronize(const int begin, const int end) const{
    for (int i = max(0, begin); i <= min(cudaStreams_num -1, end); i++)
        cudaStreamSynchronize(cudaStreams[i]);
}
