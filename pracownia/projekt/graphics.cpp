
#include "common.hpp"
#include "graphics.hpp"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

namespace hull {
  namespace graphics {

    int height = 600;
    int width = 600;
    
    GLuint myVBO;
    struct cudaGraphicsResource * myRes;
    
    void initGlWindow(int argc, char ** argv){ 

      cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

      if( !glfwInit() )
	{
	  cerr << "Failed to initalize GLFW" << endl;
	  exit( 1 );
	}

      if ( !glfwOpenWindow( width, height, 
			    8, 8, 8, 8, 8, 8,
			    GLFW_WINDOW ) )
	{
	  cerr << "Failed to open window" << endl;
	  exit( 1 );
	}

      glewInit();
      if (! glewIsSupported("GL_VERSION_2_0 ")) 
	{
	  cerr << "ERROR: Support for necessary OpenGL extensions missing." << endl;
	  exit( 1 );
	}
    
      glEnable(GL_POINT_SMOOTH);

      glViewport(0, 0, height, width);
      glLoadIdentity();
      glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

      createVBO( &myVBO, &myRes, cudaGraphicsMapFlagsWriteDiscard );

    };
  
    void closeGlWindow(){
    
      glfwCloseWindow();
      glfwTerminate();

      deleteVBO(&myVBO, myRes);

    };

    ////////////////////////////////////////////////////////////////////////////////
    //! Create VBO
    ////////////////////////////////////////////////////////////////////////////////
    void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
		   unsigned int vbo_res_flags)
    {
      int mesh_width = 100;
      int mesh_height = 100;

      // create buffer object
      glGenBuffers(1, vbo);
      glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	
      // initialize buffer object
      unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
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


  } // namespace graphics
} // namespace hull

