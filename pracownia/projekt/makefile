CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK
CUDA_PLATFORM ?= $(shell uname -m)

FILES := makefile main.cpp
LIBS := -lGL -lglfw -lboost_random -lGLEW -lcurand
COMPUTE := sm_11
GENCODE := 	-gencode arch=compute_10,code=sm_10 \
		-gencode arch=compute_10,code=sm_11 \
		-gencode arch=compute_10,code=sm_12 \
		-gencode arch=compute_10,code=sm_13 \
		-gencode arch=compute_11,code=sm_11 \
		-gencode arch=compute_11,code=sm_12 \
		-gencode arch=compute_11,code=sm_13 \
		-gencode arch=compute_12,code=sm_12 \
		-gencode arch=compute_12,code=sm_13 \
		-gencode arch=compute_13,code=sm_13 \
		-gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 

KEEP := --keep --keep-dir keep

PTX := --ptxas-options '--verbose -O4'

OBJ = obj/

O_FILES := $(addprefix $(OBJ),$(patsubst %.cpp,%.o,$(wildcard *.cpp)))

NVCC_FLAGS = -O3 -use_fast_math ${LIBS} ${GENCODE} -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} --compiler-options -mtune=native,-march=native,-ggdb3

all: $(O_FILES)
	nvcc ${NVCC_FLAGS} *.cu $(O_FILES) -o hull

$(OBJ)%.o: %.cpp
	nvcc -c ${NVCC_FLAGS} $< -o $@


kno:
	7z -tzip a kno_krzysztof_skrzetnicki_projekt_`date +%s`.zip ${FILES}

clean:
	rm -vf *.o obj/* *.hi a.out *~ hull

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs