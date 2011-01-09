
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
  boost::uniform_real<> minus_one_to_one( -1, 1 );
  boost::variate_generator< RNGType, boost::uniform_real<> > random_coord( rng, minus_one_to_one );

  typedef struct Point {
    double x;
    double y;
  } Point;

  Point random_point();
  
  void draw_point( const Point & p );
  void draw_point( const std::vector< Point > & vp );
  
  void calculateConvexHull( int n_points );

}
}
}
