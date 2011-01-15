
#include "common.hpp"
#include "hull.hpp"
#include "graphics.hpp"

#include <boost/lexical_cast.hpp>

int main(int argc, char ** argv)
{
  if ( argc < 3 )
    {
      std::cerr << "Za mało argumentów. Oczekiwano minimum 2, było: " << argc-1 << endl;
      return 1;
    }

  Processor proc;

  if ( string("cpu") == string(argv[1]) )
    {
      proc = CPU;
    }
  else if ( string("gpu") == string(argv[1]) )
    {
      proc = GPU;
    }
  else
    {
      std::cerr << "Nie podano poprawnego procesora: " << string(argv[1]) << endl;
      return 1;
    }

  vector< int > num_points;
  for(int i = 2; i < argc; i++)
    { 
      using boost::lexical_cast;
      using boost::bad_lexical_cast;

      try
	{
	  num_points.push_back(lexical_cast< int >(argv[i]));
	}
      catch(bad_lexical_cast &)
        {
	  cerr << __FILE__ << " " << __LINE__ << " " << "Zły argument: " << string(argv[i]) << endl;
	  exit(1);
        }
    }

  hull::graphics::initGlWindow(argc, argv);

  // are we finished?	
  //run = loopMode && !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );

  hull::alg::calculateConvexHull( proc, num_points );

  hull::graphics::closeGlWindow();

  return 0;

}

