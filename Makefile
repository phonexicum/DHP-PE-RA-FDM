superPrac2: main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h
	mpicxx -o superPrac2 main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -Wall -std=gnu++98 -Wno-unknown-pragmas
	mkdir -p output

superPrac2-omp: main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h
	mpicxx -o superPrac2-omp main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -Wall -std=gnu++98 -fopenmp
	mkdir -p output

.PHONY: clean

run:
	mpirun -np 2 ./superPrac2 100 0.0001 output

omprun:
	mpirun -np 2 -env OMP_NUM_THREADS=2 ./superPrac2-omp 100 0.0001 output

graph:
	./generate_gnuplot.py output
	./gnuplot.script

clean:
	rm -f -R superPrac2 superPrac2-omp output*



jmount:
	mkdir -p ./mount/
	sshfs -o nonempty edu-cmc-stud16-621-02@bluegene.hpc.cs.msu.ru:/home/edu-cmc-stud16-621-02 ./mount/

lmount:
	mkdir -p ./mount/
	sshfs -o nonempty avasilenko2_1854@lomonosov.parallel.ru:/mnt/data/users/dm4/vol12/avasilenko2_1854 ./mount/



jcompile:
	mpicxx main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -o superPrac2 -std=gnu++98 -Wall -Wno-unknown-pragmas

jcompile-omp:
	mpixlcxx_r main.cpp DHP_PE_RA_FDM.cpp -o superPrac2-omp -qsmp=omp

lcompile:
	# module add slurm/15.08
	# module add impi/5.0.1
	mpicxx main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -o superPrac2 -std=c++0x -Wall -Wno-unknown-pragmas

	cp ./superPrac2 ~/_scratch/superPrac2



upload:
	cp Makefile ./mount/

	cp main.cpp ./mount/
	cp DHP_PE_RA_FDM.cpp ./mount/
	cp DHP_PE_RA_FDM.h ./mount/



jprepare:
	mkdir -p ./output/bgp-out-1-1000
	mkdir -p ./output/bgp-out-1-2000

	mkdir -p ./output/bgp-out-128-1000
	mkdir -p ./output/bgp-out-256-1000
	mkdir -p ./output/bgp-out-512-1000

	mkdir -p ./output/bgp-out-128-2000
	mkdir -p ./output/bgp-out-256-2000
	mkdir -p ./output/bgp-out-512-2000

	mkdir -p ./output/bgp-out-128-1000-omp
	mkdir -p ./output/bgp-out-256-1000-omp
	mkdir -p ./output/bgp-out-512-1000-omp

	mkdir -p ./output/bgp-out-128-2000-omp
	mkdir -p ./output/bgp-out-256-2000-omp
	mkdir -p ./output/bgp-out-512-2000-omp

lprepare:
	mkdir -p ./_scratch/output/lom-out-1-1000
	mkdir -p ./_scratch/output/lom-out-1-2000

	mkdir -p ./_scratch/output/lom-out-8-1000
	mkdir -p ./_scratch/output/lom-out-16-1000
	mkdir -p ./_scratch/output/lom-out-32-1000
	mkdir -p ./_scratch/output/lom-out-64-1000
	mkdir -p ./_scratch/output/lom-out-128-1000

	mkdir -p ./_scratch/output/lom-out-8-2000
	mkdir -p ./_scratch/output/lom-out-16-2000
	mkdir -p ./_scratch/output/lom-out-32-2000
	mkdir -p ./_scratch/output/lom-out-64-2000
	mkdir -p ./_scratch/output/lom-out-128-2000
