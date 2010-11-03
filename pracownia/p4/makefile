

CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C

all:
	nvcc --debug -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_$(shell uname -m) OdlegloscEdycyjna.cu