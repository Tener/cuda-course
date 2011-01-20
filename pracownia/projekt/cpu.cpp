
#include "common.hpp"
#include "cpu.hpp"

#include <algorithm>
#include <set>

namespace hull {
namespace alg {
namespace cpu {

#ifdef BOOST_RANDOM
  Point random_point(){ 
    // http://www.anderswallin.net/2009/05/uniform-random-points-in-a-circle-using-polar-coordinates/
    // http://www.comnets.uni-bremen.de/itg/itgfg521/per_eval/circle_uniform_distribution.pdf
    double r = sqrt( random_radius() );
    double theta = random_radius() * 2 * M_PI;

    // http://en.wikipedia.org/wiki/Polar_coordinate_system
    double x = r * cos(theta);
    double y = r * sin(theta);

    return Point( x, y );
  }
#else
  Point random_point(){ 
    static bool was_init = false; if (!was_init) { srand48( time( 0 ) ); was_init = true; }

    double r = sqrt( drand48() );
    double theta = drand48() * 2 * M_PI;

    double x = r * cos(theta);
    double y = r * sin(theta);

    return Point( x, y );
  }
#endif

  void draw_point( const Point & p, GLfloat size = 1.0 )
  {
#ifdef USE_OPENGL
    glPointSize(size);
    glBegin(GL_POINTS);
    {
      glVertex2d( p.x, p.y );
    }
    glEnd();
#endif
  }

  void draw_point( const std::vector< Point > & vp, GLfloat size = 1.0 )
  {
#ifdef USE_OPENGL
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
#endif
  }  

  inline double cross( const Point &o, const Point &a, const Point &b )
  {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  inline double cross( const  thrust::tuple< float, float >  &o, const  thrust::tuple< float, float >  &a, const  thrust::tuple< float, float >  &b )
  {
    return (thrust::get<0>(a) - thrust::get<0>(o)) * (thrust::get<1>(b) - thrust::get<1>(o)) - (thrust::get<1>(a) - thrust::get<1>(o)) * (thrust::get<0>(b) - thrust::get<0>(o));
  }


  vector<Point> convex_hull(vector<Point> P)
  {
    int n = P.size(), k = 0;
    vector<Point> H(2*n);
    
    // Sort points lexicographically
    sort(P.begin(), P.end());
    
    // Build lower hull
    for (int i = 0; i < n; i++) {
      while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
      while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    H.resize(k);
    return H;
  }

  thrust::host_vector<Point> convex_hull(thrust::host_vector<Point> & P)
  {
    int n = P.size(), k = 0;
    vector<Point> H(2*n);
    
    // Sort points lexicographically
    sort(P.begin(), P.end());
    
    // Build lower hull
    for (int i = 0; i < n; i++) {
      while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
      while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    H.resize(k);
    return H;
  }

  std::vector< thrust::tuple< float, float > > convex_hull(thrust::host_vector< thrust::tuple< float, float > > & P)
  {
    int n = P.size(), k = 0;
    vector< thrust::tuple< float, float > > H(2*n);
    
    // Sort points lexicographically
    sort(P.begin(), P.end());
    
    // Build lower hull
    for (int i = 0; i < n; i++) {
      while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    // Build upper hull
    for (int i = n-2, t = k+1; i >= 0; i--) {
      while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
      H[k++] = P[i];
    }
    
    H.resize(k);
    return H;
  }

  void calculateConvexHull( vector< int > n_points )
  {
    for(vector<int>::iterator it = n_points.begin(); it < n_points.end(); it++)
      {
	calculateConvexHull(*it);
      }
  }

  void calculateConvexHull( int n_points )
  {
    std::vector< Point > points(1024 * n_points);
    
    for(int i = 0; i < n_points*1024; i++)
	{
	  points[i] = random_point();
	}
    
    // make points unique
    {
      std::set< Point > s( points.begin(), points.end() );
      points = std::vector< Point >( s.begin(), s.end() );
    }
    
    // draw initial points
    {
      glClear(GL_COLOR_BUFFER_BIT);
      glColor3f( 0.1, 0.3, 0.3 );
      draw_point(points);	    
    }
    
    glColor3f( 0.9, 0.6, 0.6 );
    draw_point(convex_hull(points), 5);

    glfwSwapBuffers();

  }
  
} // cpu
} // alg
} // hull
