

CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C
CUDA_PLATFORM ?= $(shell uname -m)

all:
	nvcc -DPORCELAIN_ONLY=1 -O3 -code=compute_11 -arch=compute_11 -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} OdlegloscEdycyjna.cu

test:
	ghc -O3 --make tester

rs:
	rsync --delete -avz -e "ssh -C -p 8889" OdlegloscEdycyjna.cu makefile tester.hs localhost:/home/tener/cuda/p4


kno:
	7z -tzip a kno_krzysztof_skrzetnicki_p4_`date +%s`.zip makefile OdlegloscEdycyjna.cu UWAGI.txt

