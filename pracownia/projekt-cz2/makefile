CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK
CUDA_PLATFORM ?= $(shell uname -m)

FILES := makefile main.cpp
FTGL :=  $(shell freetype-config --cflags) -lftgl
LIBS := -L${CUDA_SDK}/C/lib -lcurand -lcutil_${CUDA_PLATFORM} -lparamgl_${CUDA_PLATFORM} -lglut -lGL -lglfw -lGLEW -lpthread -lboost_system -lboost_thread-mt
#LIBS := -L${CUDA_SDK}/C/lib -lcurand -lcutil_${CUDA_PLATFORM} -lparamgl_${CUDA_PLATFORM} -lglut -lGL -lglfw -lGLEW -lpthread -Llib -lboost_system -lboost_thread
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
GENCODE_F :=
GENCODE_F := -gencode arch=compute_20,code=sm_21

# boost random
BOOST_RANDOM_FLAGS := -lboost_random -DBOOST_RANDOM
BOOST_RANDOM_FLAGS := 

# run visualization?
OPENGL := -DUSE_OPENGL 
OPENGL := 

PTX := --ptxas-options '--verbose -O4'
PTX := 
KEEP := --keep --keep-dir keep
KEEP :=

OBJ = obj/
O_FILES := $(addprefix $(OBJ),$(patsubst %.cpp,%.o,$(wildcard *.cpp)))

INCLUDES := -I${CUDA_SDK}/C/common/inc -I${CUDA_SDK}/shared/inc/ -I.
GCC_FLAGS := --compiler-options -mtune=native,-march=native
FLOAT_PREC_OPTS := -use_fast_math --prec-sqrt=true --prec-div=true
NVCC_FLAGS := -O3 ${FLOAT_PREC_OPTS} ${KEEP} ${LIBS} ${GENCODE_F} ${INCLUDES} ${GCC_FLAGS} ${OPENGL} ${BOOST_RANDOM_FLAGS} ${PTX}

rt: $(O_FILES)
	nvcc ${NVCC_FLAGS} *.cu $(O_FILES) -o rt

$(OBJ)%.o: %.cpp
	nvcc -c ${NVCC_FLAGS} $< -o $@

bench: rt
	@echo "SHA1: " bench/`sha1sum makefile | cut -d " " -f 1` 
	mkdir -p bench/`sha1sum makefile | cut -d " " -f 1`
	./rt gpu `cat benchset.txt` > benchres.txt
	cp benchset.txt benchres.txt makefile rt bench/`sha1sum makefile | cut -d " " -f 1`/

kno:
	rm -f obj/*
#	7z -tzip a projekt-cz1-krzysztof.skrzetnicki_`date +%s`.zip *.hpp *.cpp *.cu makefile thrust obj
	7z -mx=9 a dist/projekt-cz1-krzysztof.skrzetnicki_`date +%s`.7z *.hpp *.cpp *.cu makefile thrust obj

clean:
	rm -vf *.o obj/* *.hi a.out *~ rt

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs bench rt