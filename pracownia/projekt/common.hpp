#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cstdio>
#include <GL/glew.h>
#include <GL/glut.h>
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

#endif
