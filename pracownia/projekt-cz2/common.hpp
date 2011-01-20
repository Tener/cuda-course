#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cstdio>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glfw.h>

#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <string>
#include <vector>
#include <iostream>

using namespace std;

typedef enum Processor { CPU = 1, GPU = 2 } 
Processor;

extern "C" void launch_kernel_random_points(float4* vbo1, int* vbo1_vert_cnt,
					    float4* vbo2, int* vbo2_vert_cnt,
					    unsigned int points);


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error (%d) at %s:%d\n", __FILE__,__LINE__);	\
      return ;}} while(0)


#endif
