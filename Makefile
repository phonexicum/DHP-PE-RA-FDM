INCLUDES=$(wildcard *.h)
SOURCES=$(wildcard *.cpp)
CUSOURCES=$(wildcard *.cu)

# module add slurm
# module add impi/5.0.1
# module add cuda/5.5
lcompile:
	nvcc -ccbin mpicxx $(SOURCES) $(CUSOURCES) -o superPrac2 -Xcompiler -Wall
	cp ./superPrac2 ~/_scratch/superPrac2



.PHONY: clean graph lmount upload

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

mount/%.h: %.h
	cp $(@F) ./mount/

mount/%.cpp: %.cpp
	cp $(@F) ./mount/

mount/%.cu: %.cu
	cp $(@F) ./mount/

upload: $(MINCLUDES) $(MSOURCES) $(MCUSOURCES) mount/Makefile



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
