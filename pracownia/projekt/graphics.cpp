
#include "common.hpp"
#include "graphics.hpp"

namespace hull {
namespace graphics {

  int height = 600;
  int width = 600;
    
  void initGlWindow(int argc, char ** argv){ 

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

    glViewport(0, 0, height, width);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    
  };
  
  void closeGlWindow(){
    
    glfwCloseWindow();
    glfwTerminate();

  }; 

} // namespace graphics
} // namespace hull

