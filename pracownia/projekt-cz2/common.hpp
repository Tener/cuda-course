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

#include "view.h"
#include "surf.h"

#define MAX_DEBUG_STEPS 150

extern "C" void launch_debug_kernel(float4 * vbo, unsigned int * draw_cnt, View view, int w, int h);
extern "C" void launch_raytrace_kernel(uint * pbo, View view, int w, int h);



#define SERV_PORT 4000

#endif

