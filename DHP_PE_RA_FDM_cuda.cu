#include "DHP_PE_RA_FDM.h"
#include "cuda_utils.h"


// ==================================================================================================================================================
//                                                                                                                     GridDistribute::GridDistribute
// ==================================================================================================================================================
GridDistribute::GridDistribute (const cudaDeviceProp& devProp, const int tasksNum) {

    int maxThreads = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;

    tasksPerThread = (tasksNum -1) / maxThreads +1;

    demandedThreadsNumber = (tasksNum -1) / tasksPerThread +1;

    // I exceed number of registers, though I have to reduce amount of threads per block
    int availableThreadsPerBlock = devProp.maxThreadsPerBlock / 2;
    demandedThreadsPerBlock = min(demandedThreadsNumber, availableThreadsPerBlock);

    blockDim = dim3(demandedThreadsPerBlock);

    demandedBlocksNumber = (demandedThreadsNumber -1) / blockDim.x +1;

    gridDim = dim3(demandedBlocksNumber);
}


// ==================================================================================================================================================
//                                                                                                           DHP_PE_RA_FDM::cudaAllStreamsSynchronize
// ==================================================================================================================================================
void DHP_PE_RA_FDM::cudaAllStreamsSynchronize(const int begin, const int end) const{
    for (int i = max(0, begin); i <= min(cudaStreams_num -1, end); i++)
        SAFE_CUDA(cudaStreamSynchronize(cudaStreams[i]));
}
