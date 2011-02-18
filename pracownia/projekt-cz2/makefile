CUDA_SDK ?= /home/tener/localopt/NVIDIA_GPU_Computing_SDK
CUDA_PLATFORM ?= $(shell uname -m)

LIBS := -L${CUDA_SDK}/C/lib -lglut -lGL -lglfw -lGLEW -lpthread -lboost_system -lboost_thread-mt  $(shell libpng-config --ldflags)
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

PTX := 
PTX := --ptxas-options '--verbose -O4'
KEEP :=
KEEP := --keep --keep-dir keep

OBJ = obj/
O_FILES := $(addprefix $(OBJ),$(patsubst %.cpp,%.o,$(wildcard *.cpp)))

INCLUDES := -I${CUDA_SDK}/C/common/inc -I${CUDA_SDK}/shared/inc/ -I. $(shell libpng-config --cflags)
SLAVE_FLAGS := --compiler-options -mtune=native,-march=native,-O3
FLOAT_PREC_OPTS := -use_fast_math --prec-sqrt=false --prec-div=false
NVCC_FLAGS := -O3 ${FLOAT_PREC_OPTS} ${KEEP} ${LIBS} ${GENCODE_F} ${INCLUDES} ${SLAVE_FLAGS} ${OPENGL} ${BOOST_RANDOM_FLAGS} ${PTX}

rt: $(O_FILES)
	nvcc ${NVCC_FLAGS} *.cu $(O_FILES) -o rt

movies:
	find movie/ -name "0000000000.png" -not -execdir find session.avi -quit \; -execdir mencoder -mf fps=10 'mf://*.png' -ovc lavc -lavcopts threads=8:vbitrate=8000 -o session.avi \;

movies-play:
	find movie/ -name "*.avi" -exec mplayer {} +

movies-delete:
	find movie/ -name "session.avi" -print -delete

$(OBJ)%.o: %.cpp
	nvcc -c ${NVCC_FLAGS} $< -o $@

bench: rt
	@echo "SHA1: " bench/`sha1sum makefile | cut -d " " -f 1` 
	mkdir -p bench/`sha1sum makefile | cut -d " " -f 1`
	./rt gpu `cat benchset.txt` > benchres.txt
	cp benchset.txt benchres.txt makefile rt bench/`sha1sum makefile | cut -d " " -f 1`/

kno:
	rm -f obj/*
	7z -mx=9 a dist/projekt-cz2-krzysztof.skrzetnicki_`date +%s`.7z *.hpp *.cpp *.cu makefile thrust obj keep lib pdf

clean:
	rm -vf *.o obj/* *.hi a.out *~ rt

purge: clean
	rm -vf kno_*.zip

.PHONY: clean purge kno rs bench rt