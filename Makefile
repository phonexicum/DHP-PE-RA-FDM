INCLUDES=$(wildcard *.h)
SOURCES=$(wildcard *.cpp)
CUSOURCES=$(wildcard *.cu)

# module add slurm
# module add impi/5.0.1
# module add cuda/5.5
lcompile: $(INCLUDES) $(SOURCES) $(CUSOURCES)
	nvcc -rdc=true -arch=sm_20 -ccbin mpicxx $(SOURCES) $(CUSOURCES) -o superPrac2-cuda -Xcompiler -Wall -Xcompiler -O3
	cp ./superPrac2-cuda ~/_scratch/superPrac2-cuda
	cp ./profiler.config ~/_scratch/profiler.config


.PHONY: clean graph upload lmount lcompile lprepare

clean:
	rm -rf superPrac2 output* *.o



graph:
	./generate_gnuplot.py output
	./gnuplot.script



lmount:
	mkdir -p ./mount/
	sshfs -o nonempty avasilenko2_1854@lomonosov.parallel.ru:/mnt/data/users/dm4/vol12/avasilenko2_1854/cuda ./mount/



MINCLUDES=$(INCLUDES:%=mount/%)
MSOURCES=$(SOURCES:%=mount/%)
MCUSOURCES=$(CUSOURCES:%=mount/%)

mount/Makefile: Makefile
	cp Makefile ./mount/

mount/profiler.config: profiler.config
	cp profiler.config ./mount/

mount/%.h: %.h
	cp $(@F) ./mount/

mount/%.cpp: %.cpp
	cp $(@F) ./mount/

mount/%.cu: %.cu
	cp $(@F) ./mount/

upload: $(MINCLUDES) $(MSOURCES) $(MCUSOURCES) mount/Makefile mount/profiler.config



lprepare:
	mkdir -p ~/_scratch/output/lom-out-1-1000
	mkdir -p ~/_scratch/output/lom-out-1-2000

	mkdir -p ~/_scratch/output/lom-out-2-1000
	mkdir -p ~/_scratch/output/lom-out-2-2000
	mkdir -p ~/_scratch/output/lom-out-3-1000
	mkdir -p ~/_scratch/output/lom-out-3-2000
	mkdir -p ~/_scratch/output/lom-out-4-1000
	mkdir -p ~/_scratch/output/lom-out-4-2000
	mkdir -p ~/_scratch/output/lom-out-5-1000
	mkdir -p ~/_scratch/output/lom-out-5-2000
	mkdir -p ~/_scratch/output/lom-out-6-1000
	mkdir -p ~/_scratch/output/lom-out-6-2000
	mkdir -p ~/_scratch/output/lom-out-7-1000
	mkdir -p ~/_scratch/output/lom-out-7-2000
	mkdir -p ~/_scratch/output/lom-out-8-1000
	mkdir -p ~/_scratch/output/lom-out-8-2000

	mkdir -p ~/_scratch/output/lom-out-8-1000
	mkdir -p ~/_scratch/output/lom-out-16-1000
	mkdir -p ~/_scratch/output/lom-out-32-1000
	mkdir -p ~/_scratch/output/lom-out-64-1000
	mkdir -p ~/_scratch/output/lom-out-128-1000

	mkdir -p ~/_scratch/output/lom-out-8-2000
	mkdir -p ~/_scratch/output/lom-out-16-2000
	mkdir -p ~/_scratch/output/lom-out-32-2000
	mkdir -p ~/_scratch/output/lom-out-64-2000
	mkdir -p ~/_scratch/output/lom-out-128-2000
