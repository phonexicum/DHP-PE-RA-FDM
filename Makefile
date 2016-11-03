superPrac2: main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h
	mpicxx -o superPrac2 main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -Wall -std=c++11 -fopenmp

.PHONY: clean

run:
	mkdir -p output
	mpirun -np 4 ./superPrac2 50 0.1 output

graph:
	./generate_gnuplot.py output
	./gnuplot.script

clean:
	rm -f -R superPrac2 output*


jmount:
	mkdir -p ../mount/
	sshfs -o nonempty edu-cmc-stud16-621-02@bluegene.hpc.cs.msu.ru:/home/edu-cmc-stud16-621-02 ../mount/

lmount:
	mkdir -p ../mount/
	sshfs -o nonempty avasilenko2_1854@lomonosov.parallel.ru:/mnt/data/users/dm4/vol12/avasilenko2_1854 ../mount/



jcompile:
	mpicxx main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -o superPrac2 -std=gnu++98 -Wall

lcompile:
	module add slurm
	module add impi/5.0.1
	mpicxx main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM.h -o superPrac2 -std=c++0x -Wall

	cp ./superPrac2 ~/_scratch/superPrac2



upload:
	cp Makefile ../mount/

	cp main.cpp ./mount/
	cp DHP_PE_RA_FDM.cpp ./mount/
	cp DHP_PE_RA_FDM.h ./mount/



jprepare:
	mkdir -p ./_scratch/output-128-1000
	mkdir -p ./_scratch/output-256-1000
	mkdir -p ./_scratch/output-512-1000
	mkdir -p ./_scratch/output-128-2000
	mkdir -p ./_scratch/output-256-2000
	mkdir -p ./_scratch/output-512-2000
