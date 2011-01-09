
#include "common.hpp"
#include "cpu.hpp"

namespace hull {
namespace alg {
namespace cpu {

  Point random_point(){ return { random_coord(), random_coord() }; }

  void draw_point( const Point & p )
  {
    glBegin(GL_POINTS);
    {
      glVertex2d( p.x, p.y );
    }
    glEnd();
  }

  void draw_point( const std::vector< Point > & vp )
  {
    glBegin(GL_POINTS);
    {
      for(std::vector< Point >::const_iterator it = vp.begin(); 
	  it != vp.end(); 
	  it++)
	{
	  glVertex2d( it->x, it->y );
	}
    }
    glEnd();
  }  
  
  void calculateConvexHull( int n_points )
  {
    std::vector< Point > points(1000 * n_points);
    
    int run = true;

    while( run )
      {

	for(int i = 0; i < n_points; i++)
	  for(int j = 0; j < 1000; j++)
	    {
	      points[i * 1000 + j] = random_point();
	    }
	
	// draw initial points
	  {
	    glClear(GL_COLOR_BUFFER_BIT);
	    draw_point(points);	    
	    glfwSwapBuffers();
	  }

	  run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
      }
  }

} // cpu
} // alg
} // hull
