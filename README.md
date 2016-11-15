`mpi-cuda` branch is created to write DHP-PE-RA-FDM algorithm with cuda acceleration for ***Tesla X2070*** videocards under lomonosov supercomputer. This code probably is not really flexible for various video-cards (never testeds).

# DHP_PE_RA_FDM

***DHP_PE_RA_FDM*** - Dirichlet-Problem-Poisson's-Equation-Rectangular-Area-Finite-Difference-Method

This algorithm solves the Dirichlet problem for Poisson's equation in rectangular area using "steep descent iterations" for first several iterations and "conjugate gragient iterations" afterwards.

Method keypoints:

- **five-point** difference equation for Laplace operator approximation
- grid fragmentation are **regular**
- **MPI** technology for counting under supercomputers
- **CUDA** technology for NVIDIA Tesla videocard
- scalar product: (a, b) = \sum_{i=1}^{i=n-1} ( \sum_{j=1}^{j=m-1} ( h'i * h'j * a(i, j) * b(i, j) ))

Algorithm parameters:

- boundary conditions: function 'fi'
- right side of Laplace operator: function 'F'
- stopping criteria: function 'StopCriteria'

Usage:

- you have to inherit from class `DHP_PE_RA_FDM` and rewrite realization for algorithm parameters (`fi`, `F`, `stopCriteria` functions).

C++ exceptions:

- exception class `DHP_PE_RA_FDM_Exception` is successor of `std::exception`
- exception can be thrown by any process
- exist some warranties that errors concering algorithm (not MPI errors) will result in throwing exception by each involved into computations process (processes with rank [0, x_proc_num * y_proc_num])

C++ version:

- `-std=gnu++98`

Considered supercomputers:

- Lomonosov (Intel Xeon X5570 2.93GHz) (Tesla X2070)

Example:

- example, realized in main benchmark is 2-nd variant in task description in `algorithm.pdf` on page 6
- for mentioned example etalon graph and computed graph were generated

## Repository structure

- `Makefile` - contains a lot of instructions for mounting, uploading, compiling, etc. under supercomputers and local machine

- `./main.cpp` - C++ file containing benchmark for algorithm
- `./*.h`, `./*.cpp` and `./*.cu` - C++ and CUDA realization of DHP_PE_RA_FDM

- `./generate_gnuplot.py` - generates `./*.dat` files for their plotting
- `./gnuplot.script` - gnuplot script which generates plots

- `./algorithm/` - problem formulation (*russian only*) (contains calculating formules) and scheme of numerical method
- `./loutput/` - directories contain some results of running benchmark under Lomonosov supercomputers

## Lomonosov videocard technical specification

cudaDeviceProp
```
deviceNum= 2
videocard-name= Tesla X2070
cudaVersion= 2.0

multiProcessorCount= 14
maxThreadsPerMultiProcessor= 1536
warpSize= 32

maxThreadsPerBlock= 1024
maxGridSize= 65535 65535 65535
maxThreadsDim= 1024 1024 64

totalGlobalMem= 5636554752 // 5 GB
totalConstMem= 65536
sharedMemPerBlock= 49152
regsPerBlock= 32768

canMapHostMemory= 1     // Memory can be locked (disable virtual memory mechanism) - enables to copy memory from host to device through PCI making it asynchronously past CPU
unifiedAddressing= 1    // Memory allocated by cudaHostAllocate (..., cudaHostAllocMapped) automaticaly became mapped and pointer to the memory is unique for CPU and other devices
computeMode= 0          // Multiple threads can use cudaSetDevice() with this device
asyncEngineCount= 2     // 1 - device can copy and execute kernel in parallel, 2 - can copy in both directions and execute kernel
concurrentKernels= 1    // device's kernels can be computed in parallel
deviceOverlap= 1        // Device can concurrently copy memory and execute a kernel

// ================================

clockRate= 1147000

ECCEnabled= 1

integrated= 0
kernelExecTimeoutEnabled= 0

l2CacheSize= 786432

memoryBusWidth= 384
memoryClockRate= 1548000
memPitch= 2147483647

pciBusID= 3
pciDeviceID= 0
pciDomainID= 0

surfaceAlignment= 512
```

## supercomputer commands

### Lomonosov

Lomonosov execute tasks in context of `~/_scratch` directory.

```
sbatch -p gputest -n 1 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 5 0.001 output
sbatch -p gputest -n 4 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 1000 0.0001 output





sbatch -p gputest -n 1 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-1-1000 # 00:07 sec
sbatch -p gputest -n 1 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 2000 0.0001 output/lom-out-1-2000 # 00:56 sec

sbatch -p gputest -n 8 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 1000 0.0001 output/lom-out-8-1000 # 00:04 sec
sbatch -p gputest -n 8 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 2000 0.0001 output/lom-out-8-2000 # 00:13 sec
sbatch -p gputest -n 16 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 1000 0.0001 output/lom-out-16-1000 # 00:05 sec
sbatch -p gputest -n 16 --ntasks-per-node=2 --time=0-00:15:00 impi ./superPrac2 2000 0.0001 output/lom-out-16-2000 # 00:12 sec

sbatch -p gpu -n 32 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-32-1000 #
sbatch -p gpu -n 32 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 2000 0.0001 output/lom-out-32-2000 #
sbatch -p gpu -n 64 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-64-1000 #
sbatch -p gpu -n 64 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 2000 0.0001 output/lom-out-64-2000 #
sbatch -p gpu -n 128 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-128-1000 #
sbatch -p gpu -n 128 --ntasks-per-node=2 --time=0-00:01:00 impi ./superPrac2 2000 0.0001 output/lom-out-128-2000 #


```
