
#include "common.hpp"
#include "graphics.hpp"

#include <cutil_gl_error.h>

namespace rt {
  namespace graphics {
    VBO::VBO(unsigned int n_points, unsigned int vbo_res_flags)
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

    VBO::~VBO()
    {
      // unregister this buffer object with CUDA
      cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	
      glBindBuffer(1, vbo); // is this really needed???
      glDeleteBuffers(1, &vbo);
	  
      vbo = 0;
    }

    void VBO::render( int point_cnt, float3 color )
    {
      if ( point_cnt == -1 )
        point_cnt = n_points;
        

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glVertexPointer(4, GL_FLOAT, 0, 0);

      glEnableClientState(GL_VERTEX_ARRAY);
      glColor3f(color.x, color.y, color.z);
      glDrawArrays(GL_POINTS, 0, point_cnt);
      glDisableClientState(GL_VERTEX_ARRAY);

    }

    float4 * VBO::mapResourcesGetMappedPointer()
    {
      float4 *dptr;
      size_t num_bytes; 

      cutilSafeCall(cudaGraphicsMapResources(1, &cuda_vbo_resource));
      cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,  
                                                         cuda_vbo_resource));

      return dptr;
    }

    void VBO::unmapResources()
    {
      cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }
  

    PBO::PBO(int p_width, int p_height) : width(p_width), height(p_height), pbo(0)
    {
      glGenBuffersARB(1, &pbo);
      glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
      glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
      cutilSafeCall(cudaGLRegisterBufferObject(pbo));
    }

    void PBO::mapBufferObject()
    {
        cudaGLMapBufferObject((void**)&dev_pbo, pbo);
    }
    
    void PBO::unmapBufferObject()
    {
      cudaGLUnmapBufferObject(pbo);
    }

    void PBO::render()
    {
      glClear(GL_COLOR_BUFFER_BIT);
      glDisable(GL_DEPTH_TEST);
      glRasterPos2i(0, 0);
      glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }

    GLManager global_glm;

    void GLManager::initGlWindow(){ 
      cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

      if( !glfwInit() )
	{
	  std::cerr << "Failed to initalize GLFW" << std::endl;
	  exit( 1 );
	}

      glfwOpenWindowHint( GLFW_WINDOW_NO_RESIZE, GL_TRUE );

      if ( !glfwOpenWindow( width, height, 
			    8, 8, 8, 8, 8, 8,
			    GLFW_WINDOW ) )
	{
	  std::cerr << "Failed to open window" << std::endl;
	  exit( 1 );
	}

      glewInit();
      if (! glewIsSupported("GL_VERSION_2_0 ")) 
	{
	  std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
	  exit( 1 );
	}
    


      glViewport(0, 0, height, width);
      glLoadIdentity();
      //      glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    };
  
    void GLManager::closeGlWindow(){
      glfwCloseWindow();
      glfwTerminate();
    };

    void GLManager::reshape(int w, int h){ }

  } // namespace graphics
} // namespace rt

