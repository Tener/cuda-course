

CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C
CUDA_PLATFORM ?= $(shell uname -m)

ifdef tile
TILE_OPT = -DTILE=${tile}
endif

PROG_NAME ?= kernel-$(shell uname -n).out

all:
	nvcc -O3 ${TILE_OPT} -code=compute_13 -arch=compute_13 -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} OdlegloscEdycyjna.cu -o ${PROG_NAME}

test:
	ghc -O3 --make tester
	make all
	env PORCELAIN=1 SKIP_CPU=1 ./tester ./${PROG_NAME} slowa_small.txt
	env PORCELAIN=1 SKIP_CPU=1 ./tester ./${PROG_NAME} slowa.txt

# rsync with remote machine
rs:
	rsync --delete -avz -e "ssh -C -p 8889" OdlegloscEdycyjna.cu makefile tester.hs localhost:/home/tener/cuda/p4

kno:
	7z -tzip a kno_krzysztof_skrzetnicki_p4_`date +%s`.zip readme.txt OdlegloscEdycyjna.cu makefile tester.hs slowa_small.txt

tags:
	etags.emacs *.cu

clean:
	rm -vf *.o *.hi TAGS tester kernel-*.out a.out *~

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs test