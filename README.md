# DHP_PE_RA_FDM

***DHP_PE_RA_FDM*** - Dirichlet-Problem-Poisson's-Equation-Rectangular-Area-Finite-Difference-Method

This algorithm solves the Dirichlet problem for Poisson's equation in rectangular area using "steep descent iterations" for first several iterations and "conjugate gragient iterations" afterwards.

Method keypoints:

- **five-point** difference equation for Laplace operator approximation
- grid fragmentation are **regular**
- **MPI** technology for counting under supercomputers
- **OpenMP** technology for parallelization between processor cores
- scalar product: (a, b) = \sum_{i=1}^{i=n-1} ( \sum_{j=1}^{j=m-1} ( h'i * h'j * a(i, j) * b(i, j) ))

Algorithm parameters:

- boundary conditions: function 'fi'
- right side of Laplace operator: function 'F'
- stopping criteria: function 'stopCriteria'

Usage:

- you have to inherit from class `DHP_PE_RA_FDM` and write realization for algorithm parameters (virtual functions).

C++ exceptions:

- exception class `DHP_PE_RA_FDM_Exception` is successor of `std::exception`

C++ version:

- `-std=gnu++98`

Considered supercomputers:

- Lomonosov
- Blue Gene/P (g++ -std=gnu++98)

## Repository structure

- `Makefile` - contains a lot of instructions for mounting, uploading, compiling, etc. under supercomputers and local machine

- `./main.cpp` - C++ file containing benchmark for PAM algorithm
- `./DHP_PE_RA_FDM.h` and `./DHP_PE_RA_FDM.cpp` - C++ DHP_PE_RA_FDM realization

- `./generate_gnuplot.py` - generates `./*.dat` files for their plotting
- `./gnuplot.script` - gnuplot script which generates plots

- `./algorithm.pdf` - problem formulation (*russian only*) (contains calculating formules)

## supercomputer commands

### Blue Gene

```
mpisubmit.bg -n 128 -m smp -w 00:15:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-128-1000
mpisubmit.bg -n 256 -m smp -w 00:10:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-256-1000
mpisubmit.bg -n 512 -m smp -w 00:05:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-512-1000

mpisubmit.bg -n 128 -m smp -w 00:15:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-128-2000
mpisubmit.bg -n 256 -m smp -w 00:10:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-256-2000
mpisubmit.bg -n 512 -m smp -w 00:05:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-512-2000


mpisubmit.bg -n 128 -m smp -env OMP_NUM_THREADS=3 -w 00:15:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-128-1000
mpisubmit.bg -n 256 -m smp -env OMP_NUM_THREADS=3 -w 00:10:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-256-1000
mpisubmit.bg -n 512 -m smp -env OMP_NUM_THREADS=3 -w 00:05:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output-512-1000

mpisubmit.bg -n 128 -m smp -env OMP_NUM_THREADS=3 -w 00:15:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-128-2000
mpisubmit.bg -n 256 -m smp -env OMP_NUM_THREADS=3 -w 00:10:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-256-2000
mpisubmit.bg -n 512 -m smp -env OMP_NUM_THREADS=3 -w 00:05:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output-512-2000


mpisubmit.bg -n 1 -m smp -w 02:00:00 -t PREFER_TORUS
mpisubmit.bg -n 16 -m smp -w 02:00:00 -t PREFER_TORUS
mpisubmit.bg -n 64 -m smp -w 02:00:00 -t PREFER_TORUS
mpisubmit.bg -n 128 -m smp -w 00:15:00 -t PREFER_TORUS
mpisubmit.bg -n 256 -m smp -w 00:10:00 -t PREFER_TORUS
mpisubmit.bg -n 512 -m smp -w 00:05:00 -t PREFER_TORUS
mpisubmit.bg -n 1024 -m smp -w 00:02:00 -t PREFER_TORUS
```

### Lomonosov

Lomonosov execute tasks in context of `~/_scratch` directory.

```
sbatch -p regular4 -n 128 impi ./superPrac2 1000 0.0001 output-128-1000
```





