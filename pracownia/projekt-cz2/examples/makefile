
O_FILES := $(addprefix $(OBJ),$(patsubst %.cu,%.bin,$(wildcard *.cu)))

all: ${O_FILES}

$(OBJ)%.bin: %.cu
	nvcc -I.. -O3 -use_fast_math ${NVCC_FLAGS} $< -o $@


clean:
	rm -f *~ *.bin

exec:
	find -type f -executable -exec ./{} \;