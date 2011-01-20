
#include "common.hpp"
#include "graphics.hpp"

namespace hull {
  namespace graphics {

    int height = 600;
    int width = 600;
    
    void initGlWindow(int argc, char ** argv){ 
#ifdef USE_OPENGL
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
    
      //glEnable(GL_POINT_SMOOTH);

      glViewport(0, 0, height, width);
      glLoadIdentity();
      glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);
#endif
    };
  
    void closeGlWindow(){
#ifdef USE_OPENGL   
      glfwCloseWindow();
      glfwTerminate();
#endif
    };

  } // namespace graphics
} // namespace hull

