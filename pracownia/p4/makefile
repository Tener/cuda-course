

CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C
CUDA_PLATFORM ?= $(shell uname -m)

all:
	nvcc --debug -code=compute_11 -arch=compute_11 -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} OdlegloscEdycyjna.cu

kno:
	7z -tzip a kno_krzysztof_skrzetnicki_p4_`date +%s`.zip makefile OdlegloscEdycyjna.cu

