
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

      struct VBO {
	// vbo variables
	GLuint vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	void * d_vbo_buffer;
	unsigned int n_points;

	VBO(unsigned int n_points, unsigned int vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard)
	  : d_vbo_buffer(NULL), n_points(n_points)
	{
	  //GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
	  
	  // create buffer object
	  glGenBuffers(1, &vbo);
	  glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	  // initialize buffer object
	  unsigned int size = n_points * 4 * sizeof(float);
	  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
	  glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	  // register this buffer object with CUDA
	  cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, vbo_res_flags));
  
	  CUT_CHECK_ERROR_GL();
	}

	~VBO()
	{
	  // unregister this buffer object with CUDA
	  cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	
	  glBindBuffer(1, vbo); // is this really needed???
	  glDeleteBuffers(1, &vbo);
	  
	  vbo = 0;
	}

	void render( unsigned int point_cnt, float3 color )
	{
	  glBindBuffer(GL_ARRAY_BUFFER, vbo);
	  glVertexPointer(4, GL_FLOAT, 0, 0);

	  glEnableClientState(GL_VERTEX_ARRAY);
	  glColor3f(color.x, color.y, color.z);
	  glDrawArrays(GL_POINTS, 0, point_cnt);
	  glDisableClientState(GL_VERTEX_ARRAY);

	}

	float4 * mapResourcesGetMappedPointer()
	{
	  float4 *dptr;
	  size_t num_bytes; 

	  cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource));
	  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,  
							     cuda_vbo_resource));

	  return dptr;
	}

	void unmapResources()
	{
	  cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}
      };
      
      void calculateConvexHull( int n_points, VBO & vbo_1, VBO & vbo_2 )
      {
	// clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	
	// map OpenGL buffer object for writing from CUDA
	int vbo1_cnt, vbo2_cnt;

	launch_kernel_random_points(vbo_1.mapResourcesGetMappedPointer(), &vbo1_cnt,
				    vbo_2.mapResourcesGetMappedPointer(), &vbo2_cnt,
				    n_points);

	vbo_1.unmapResources();
	vbo_2.unmapResources();

	// render from the vbo
	vbo_1.render( vbo1_cnt, make_float3( 1.0, 0.0, 0.0 ) );
	vbo_2.render( vbo2_cnt, make_float3( 0.0, 0.0, 1.0 ) );

	glfwSwapBuffers();
      }

      void calculateConvexHull( vector< int > n_points )
      {
	int max_n_points = *max_element( n_points.begin(), n_points.end() );

	VBO vbo_1( max_n_points * 1024 );
	VBO vbo_2( max_n_points * 1024 );

	for(vector<int>::iterator it = n_points.begin(); it < n_points.end(); it++)
	  {
	    calculateConvexHull(*it, vbo_1, vbo_2);
	  }
      }
    }
  }
}
