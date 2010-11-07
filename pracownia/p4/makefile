

CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C
CUDA_PLATFORM ?= $(shell uname -m)

all:
	nvcc -O3 -code=compute_11 -arch=compute_11 -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} OdlegloscEdycyjna.cu

test:
	ghc -O3 --make tester
	make all
	env PORCELAIN=1 ./tester ./a.out slowa_small.txt
	env PORCELAIN=1 ./tester ./a.out slowa.txt

# rsync with remote machine
rs:
	rsync --delete -avz -e "ssh -C -p 8889" OdlegloscEdycyjna.cu makefile tester.hs localhost:/home/tener/cuda/p4

kno:
	7z -tzip a kno_krzysztof_skrzetnicki_p4_`date +%s`.zip UWAGI.txt OdlegloscEdycyjna.cu makefile tester.hs slowa_small.txt

clean:
	rm -vf *.o *.hi a.out *~

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs test