# module add slurm
# module add impi/5.0.1
# module add cuda/5.5


lcompile:
	nvcc -ccbin mpicxx main.cpp DHP_PE_RA_FDM.cpp DHP_PE_RA_FDM_cuda.cu -o superPrac2 -Xcompiler -Wall
	cp ./superPrac2 ~/_scratch/superPrac2


# SOURCES=main.cpp DHP_PE_RA_FDM.cpp
# CUSOURCES=DHP_PE_RA_FDM_kernels.cu
# INCLUDES=DHP_PE_RA_FDM.h
# OBJECTS=$(SOURCES:.cpp=.cpp.o)
# CUOBJECTS=$(CUSOURCES:.cu=.cu.o)
# EXECUTABLE=superPrac2

# lcompile: $(EXECUTABLE)

# $(EXECUTABLE): $(OBJECTS) $(CUOBJECTS)
# 	g++ -L/opt/cuda/cuda-5.5/bin/..//bin64 -lcudart $(OBJECTS) $(CUOBJECTS) -o $@
# 	cp ./$(EXECUTABLE) ~/_scratch/$(EXECUTABLE)

# %.cpp.o: %.cpp
# 	mpicxx -I/opt/cuda/cuda-5.5/bin/..//include $< $(INCLUDES) -c -o $@

# %.cu.o: %.cu
# 	nvcc -v -arch=sm_20 -Xptxas -v $< -c -o $@


.PHONY: clean

clean:
	rm -rf superPrac2 output* *.o


graph:
	./generate_gnuplot.py output
	./gnuplot.script


lmount:
	mkdir -p ./mount/
	sshfs -o nonempty avasilenko2_1854@lomonosov.parallel.ru:/mnt/data/users/dm4/vol12/avasilenko2_1854/cuda ./mount/

upload:
	cp Makefile ./mount/
	cp main.cpp ./mount/
	cp DHP_PE_RA_FDM.cpp ./mount/
	cp DHP_PE_RA_FDM.h ./mount/
	cp DHP_PE_RA_FDM_cuda.cu ./mount/

ldown:
	cp -R ~/_scratch/output/ ./output/

down:
	mkdir -p ./output/
	cp -R ./mount/output/ ./output/

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
