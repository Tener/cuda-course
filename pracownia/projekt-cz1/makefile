CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK
CUDA_PLATFORM ?= $(shell uname -m)

FILES := makefile main.cpp
LIBS := -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} -lGL -lglfw -lGLEW -lcurand
GENCODE_FULL :=	-gencode arch=compute_10,code=sm_10 \
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

GENCODE_F := -gencode arch=compute_11,code=sm_13
GENCODE_F := -gencode arch=compute_20,code=sm_21
GENCODE_F :=

# boost random
BOOST_RANDOM_FLAGS := -lboost_random -DBOOST_RANDOM
BOOST_RANDOM_FLAGS := 

# run visualization?
OPENGL := -DUSE_OPENGL 
OPENGL := 

PTX := --ptxas-options '--verbose -O4'
PTX := 
KEEP := --keep --keep-dir keep

OBJ = obj/
O_FILES := $(addprefix $(OBJ),$(patsubst %.cpp,%.o,$(wildcard *.cpp)))

INCLUDES := -I${CUDA_SDK}/C/common/inc -I${CUDA_SDK}/shared/inc/ -I.
GCC_FLAGS := --compiler-options -mtune=native,-march=native,-ffast-math
NVCC_FLAGS := -O3 -use_fast_math ${LIBS} ${GENCODE_F} ${INCLUDES} ${GCC_FLAGS} ${OPENGL} ${BOOST_RANDOM_FLAGS} ${PTX}

hull: $(O_FILES)
	nvcc ${NVCC_FLAGS} *.cu $(O_FILES) -o hull

$(OBJ)%.o: %.cpp
	nvcc -c ${NVCC_FLAGS} $< -o $@

bench: hull
	@echo "SHA1: " bench/`sha1sum makefile | cut -d " " -f 1` 
	mkdir -p bench/`sha1sum makefile | cut -d " " -f 1`
	./hull gpu `cat benchset.txt` > benchres.txt
	cp benchset.txt benchres.txt makefile hull bench/`sha1sum makefile | cut -d " " -f 1`/

kno:
	rm -f obj/*
#	7z -tzip a projekt-cz1-krzysztof.skrzetnicki_`date +%s`.zip *.hpp *.cpp *.cu makefile thrust obj
	7z -mx=9 a dist/projekt-cz1-krzysztof.skrzetnicki_`date +%s`.7z *.hpp *.cpp *.cu makefile thrust obj

clean:
	rm -vf *.o obj/* *.hi a.out *~ hull

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs bench hull