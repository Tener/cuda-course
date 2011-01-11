
#include "common.hpp"
#include "cpu.hpp"

#include <algorithm>

namespace hull {
namespace alg {
namespace cpu {

  // http://mathworld.wolfram.com/CirclePointPicking.html
  Point random_point(){ 
    double x, y;
    do {
      x = random_coord();
      y = random_coord();
    }
    while ( x*x + y*y >= 1 );
    
    return {x,y};
  }
    
#if 0
  Point random_point(){ 
    // http://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/
    // http://www.comnets.uni-bremen.de/itg/itgfg521/per_eval/circle_uniform_distribution.pdf
    double r = sqrt( random_radius() );
    double theta = random_angle();

    // http://en.wikipedia.org/wiki/Polar_coordinate_system
    double x = r * cos(theta);
    double y = r * sin(theta);
    
    cout << 
      "X:" << x << " " <<
      "Y:" << y << " " <<
      "R:" << r << " " << 
      "T:" << theta << " " <<
      endl;

    return { x, y };
  }
#endif

  void draw_point( const Point & p, GLfloat size = 1.0 )
  {
    glPointSize(size);
    glBegin(GL_POINTS);
    {
      glVertex2d( p.x, p.y );
    }
    glEnd();
  }

  void draw_point( const std::vector< Point > & vp, GLfloat size = 1.0 )
  {
    glPointSize(size);
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

	// find leftmost, downmost point
	std::vector< Point >::iterator minimal_el =  min_element( points.begin(), points.end() );

	cout << minimal_el->x << " " << minimal_el->y << endl;

	// are we finished?	
	run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
      }
  }

} // cpu
} // alg
} // hull
