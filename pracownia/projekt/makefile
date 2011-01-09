CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK
CUDA_PLATFORM ?= $(shell uname -m)

FILES := makefile main.cpp
OPENGL_LIBS := -lGL -lglfw
COMPUTE := compute_12

PTX := --ptxas-options '--verbose -O4'

all:
	nvcc -O3 -use_fast_math -code=${COMPUTE} -arch=${COMPUTE} ${OPENGL_LIBS} -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} --compiler-options -std=c++0x *.cpp

kno:
	7z -tzip a kno_krzysztof_skrzetnicki_projekt_`date +%s`.zip ${FILES}

clean:
	rm -vf *.o *.hi a.out *~

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs