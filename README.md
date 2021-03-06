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
- exception can be thrown by any process
- exist some warranties that errors concering algorithm (not MPI errors) will result in throwing exception by each involved into computations process (processes with rank [0, x_proc_num * y_proc_num])

C++ version:

- `-std=gnu++98`

Considered supercomputers:

- Lomonosov
- Blue Gene/P (g++ -std=gnu++98)

Example:

- example, realized in main benchmark is 2-nd variant in task description in `algorithm.pdf` on page 6
- for mentioned example etalon graph and computed graph were generated

## Repository structure

- `Makefile` - contains a lot of instructions for mounting, uploading, compiling, etc. under supercomputers and local machine

- `./main.cpp` - C++ file containing benchmark for algorithm
- `./DHP_PE_RA_FDM.h` and `./DHP_PE_RA_FDM.cpp` - C++ DHP_PE_RA_FDM realization

- `./generate_gnuplot.py` - generates `./*.dat` files for their plotting
- `./gnuplot.plt` - gnuplot script which generates plots

- `./algorithm/` - problem formulation (*russian only*) (contains calculating formules) and scheme of numerical method
- `./joutput/` and `./loutput/` - directories contain some results of running benchmark under BlueGene/P and Lomonosov supercomputers

## supercomputer commands

### Blue Gene

```
mpisubmit.bg -n 1 -m smp -w 02:00:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output/bgp-out-1-1000 # 50:15 sec
mpisubmit.bg -n 1 -m smp -w 02:00:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output/bgp-out-1-2000 # --:--
mpisubmit.bg -n 1 -m smp -env OMP_NUM_THREADS=3 -w 02:00:00 -t PREFER_TORUS ./superPrac2-omp 1000 0.0001 output/bgp-out-1-1000-omp # 04:22
mpisubmit.bg -n 1 -m smp -env OMP_NUM_THREADS=3 -w 02:00:00 -t PREFER_TORUS ./superPrac2-omp 2000 0.0001 output/bgp-out-1-2000-omp # 35:37

mpisubmit.bg -n 2 -m smp -w 01:00:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output/bgp-out-2-1000 # 21:37 sec
mpisubmit.bg -n 2 -m smp -w 02:00:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output/bgp-out-2-2000 # --:--
mpisubmit.bg -n 2 -m smp -env OMP_NUM_THREADS=3 -w 01:00:00 -t PREFER_TORUS ./superPrac2-omp 1000 0.0001 output/bgp-out-2-1000-omp # 02:12
mpisubmit.bg -n 2 -m smp -env OMP_NUM_THREADS=3 -w 01:00:00 -t PREFER_TORUS ./superPrac2-omp 2000 0.0001 output/bgp-out-2-2000-omp # 17:50

mpisubmit.bg -n 128 -m smp -w 00:02:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output/bgp-out-128-1000 # 0:24 sec
mpisubmit.bg -n 128 -m smp -w 00:05:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output/bgp-out-128-2000 # 3:10 sec
mpisubmit.bg -n 128 -m smp -env OMP_NUM_THREADS=3 -w 00:01:00 -t PREFER_TORUS ./superPrac2-omp 1000 0.0001 output/bgp-out-128-1000-omp # 0:04 sec
mpisubmit.bg -n 128 -m smp -env OMP_NUM_THREADS=3 -w 00:01:00 -t PREFER_TORUS ./superPrac2-omp 2000 0.0001 output/bgp-out-128-2000-omp # 0:31 sec

mpisubmit.bg -n 256 -m smp -w 00:01:00 -t PREFER_TORUS ./superPrac2 1000 0.0001 output/bgp-out-256-1000 # 0:12 sec
mpisubmit.bg -n 256 -m smp -w 00:03:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output/bgp-out-256-2000 # 1:35 sec
mpisubmit.bg -n 256 -m smp -env OMP_NUM_THREADS=3 -w 00:00:30 -t PREFER_TORUS ./superPrac2-omp 1000 0.0001 output/bgp-out-256-1000-omp # 0:03 sec
mpisubmit.bg -n 256 -m smp -env OMP_NUM_THREADS=3 -w 00:01:00 -t PREFER_TORUS ./superPrac2-omp 2000 0.0001 output/bgp-out-256-2000-omp # 0:16 sec

mpisubmit.bg -n 512 -m smp -w 00:00:30 -t PREFER_TORUS ./superPrac2 1000 0.0001 output/bgp-out-512-1000 # 0:06 sec
mpisubmit.bg -n 512 -m smp -w 00:03:00 -t PREFER_TORUS ./superPrac2 2000 0.0001 output/bgp-out-512-2000 # 1:48 sec
mpisubmit.bg -n 512 -m smp -env OMP_NUM_THREADS=3 -w 00:00:30 -t PREFER_TORUS ./superPrac2-omp 1000 0.0001 output/bgp-out-512-1000-omp # 0:2 sec
mpisubmit.bg -n 512 -m smp -env OMP_NUM_THREADS=3 -w 00:01:00 -t PREFER_TORUS ./superPrac2-omp 2000 0.0001 output/bgp-out-512-2000-omp # 0:9 sec


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
sbatch -p test -n 1 --time=0-00:07:00 impi ./superPrac2 1000 0.0001 output/lom-out-1-1000 # 5:09 sec
sbatch -p regular4 -n 1 --time=0-00:59:30 impi ./superPrac2 2000 0.0001 output/lom-out-1-2000 # 41:10 sec

sbatch -p test -n 8 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-8-1000 # 0:39 sec
sbatch -p test -n 8 --time=0-00:07:00 impi ./superPrac2 2000 0.0001 output/lom-out-8-2000 # 5:11 sec
sbatch -p test -n 16 --time=0-00:01:00 impi ./superPrac2 1000 0.0001 output/lom-out-16-1000 # 0:19 sec
sbatch -p test -n 16 --time=0-00:03:50 impi ./superPrac2 2000 0.0001 output/lom-out-16-2000 # 2:36 sec
sbatch -p test -n 32 --time=0-00:00:30 impi ./superPrac2 1000 0.0001 output/lom-out-32-1000 # 0:9 sec
sbatch -p test -n 32 --time=0-00:02:00 impi ./superPrac2 2000 0.0001 output/lom-out-32-2000 # 1:18 sec
sbatch -p test -n 64 --time=0-00:00:10 impi ./superPrac2 1000 0.0001 output/lom-out-64-1000 # 0:5 sec
sbatch -p test -n 64 --time=0-00:02:00 impi ./superPrac2 2000 0.0001 output/lom-out-64-2000 # 0:39 sec
sbatch -p test -n 128 --time=0-00:00:10 impi ./superPrac2 1000 0.0001 output/lom-out-128-1000 # 0:2 sec
sbatch -p test -n 128 --time=0-00:01:00 impi ./superPrac2 2000 0.0001 output/lom-out-128-2000 # 0:19 sec
```
