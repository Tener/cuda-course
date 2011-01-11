
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>

#include <cstdlib>
#include <time.h>

namespace hull {
namespace alg {
namespace cpu {

  typedef boost::mt19937 RNGType;

  RNGType rng( time( 0 ) ); // produces randomness out of thin air
  boost::uniform_real<> uniform_0_2pi( 0, 2 * M_PI );
  boost::uniform_real<> uniform_0_1( 0, 1 );
  boost::uniform_real<> uniform_m1_1( -1, 1 );
  boost::variate_generator< RNGType, boost::uniform_real<> > random_angle( rng, uniform_0_2pi );
  boost::variate_generator< RNGType, boost::uniform_real<> > random_radius( rng, uniform_0_1 );
  boost::variate_generator< RNGType, boost::uniform_real<> > random_coord( rng, uniform_m1_1 );

  typedef struct Point {
    double x;
    double y;

    inline bool operator< (const Point &other) const { 
      if (y < other.y)
	return true;
      if (y > other.y)
	return false;
      return x < other.x;
    }
  } Point;

  Point random_point();
  
  void draw_point( const Point & p, GLfloat size );
  void draw_point( const std::vector< Point > & vp, GLfloat size );
  
  void calculateConvexHull( int n_points );

}
}
}
