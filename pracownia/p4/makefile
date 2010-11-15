
CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK/C
CUDA_PLATFORM ?= $(shell uname -m)

ifdef tile
TILE_OPT = -DTILE=${tile}
endif

PROG_NAME := ./kernel-a.out
PROG_TESTER := ./tester

PROG_NAME ?= kernel-$(shell uname -n).out
PROG_TESTER ?= tester-$(shell uname -n)

all:
	nvcc --debug -O3 ${TILE_OPT} -code=compute_13 -arch=compute_13 -I${CUDA_SDK}/C/common/inc -L${CUDA_SDK}/C/lib -lcutil_${CUDA_PLATFORM} OdlegloscEdycyjna.cu -o ${PROG_NAME}

compile-tester:
	ghc -O3 --make tester -o ${PROG_TESTER}
	make all
test:
	make compile-tester
	${PROG_TESTER} ${PROG_NAME} slowa_small.txt
	${PROG_TESTER} ${PROG_NAME} slowa.txt

# rsync with remote machine
rs:
	rsync --delete -avz OdlegloscEdycyjna.cu makefile tester.hs slowa.txt slowa_small.txt ~/remote/ii/cuda/p4

kno:
	7z -tzip a kno_krzysztof_skrzetnicki_p4_`date +%s`.zip readme.txt OdlegloscEdycyjna.cu makefile tester.hs slowa_small.txt

tags:
	etags.emacs *.cu

clean:
	rm -vf *.o *.hi TAGS tester-* tester kernel-*.out a.out *~

purge: clean
	rm -vf kno_*.zip

dist-env:
	cd ..
	mkdir p4-shadow-$(shell uname -n)
	cd p4-shadow-$(shell uname -n)
	for f in ../p4/* do
		ln -s ../p4/$f
	done

.PHONY: clean purge kno rs test