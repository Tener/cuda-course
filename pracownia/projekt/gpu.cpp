
#include "common.hpp"
#include "gpu.hpp"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

namespace hull {
  namespace alg {
    namespace gpu {

      // vbo variables
      GLuint vbo;
      struct cudaGraphicsResource *cuda_vbo_resource;
      void *d_vbo_buffer = NULL;
  void calculateConvexHull( int n_points )
  {
  }

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
	cudaGraphicsUnregisterResource(vbo_res);
	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
      }

}
}
}
