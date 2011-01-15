
#include "common.hpp"
#include "gpu.hpp"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include <algorithm>

namespace hull {
  namespace alg {
    namespace gpu {

      // vbo variables
      GLuint vbo;
      struct cudaGraphicsResource *cuda_vbo_resource;
      void *d_vbo_buffer = NULL;

      ////////////////////////////////////////////////////////////////////////////////
      //! Create VBO
      ////////////////////////////////////////////////////////////////////////////////
      void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
		     unsigned int vbo_res_flags,
		     unsigned int n_points)
      {
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	
	// initialize buffer object
	unsigned int size = n_points * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	// register this buffer object with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
  
	CUT_CHECK_ERROR_GL();
     }

      ////////////////////////////////////////////////////////////////////////////////
      //! Delete VBO
      ////////////////////////////////////////////////////////////////////////////////
      void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res)
      {
	// unregister this buffer object with CUDA
	cutilSafeCall(cudaGraphicsUnregisterResource(vbo_res));
	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
      }

      void calculateConvexHull( vector< int > n_points )
      {
	int max_n_points = *max_element( n_points.begin(), n_points.end() );

	createVBO( &vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard, max_n_points * 1024 );

	for(vector<int>::iterator it = n_points.begin(); it < n_points.end(); it++)
	  {
	    calculateConvexHull(*it);
	  }

	deleteVBO(&vbo, cuda_vbo_resource);
      }
      
      void calculateConvexHull( int n_points )
      {
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes; 
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,  
							   cuda_vbo_resource));

	launch_kernel_random_points(dptr, n_points);
    
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// draw on screen

	glClear(GL_COLOR_BUFFER_BIT);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, n_points * 1024);
	glDisableClientState(GL_VERTEX_ARRAY);

	glfwSwapBuffers();
      }

    }
  }
}
