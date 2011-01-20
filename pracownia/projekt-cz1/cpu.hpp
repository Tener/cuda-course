
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

#ifdef BOOST_RANDOM
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#endif

#include <thrust/host_vector.h>

#include <cstdlib>
#include <time.h>

struct Point {
  double x;
  double y;

  __host__ __device__
  Point() : x(0), y(0) { }
  __host__ __device__
  Point(double x, double y) : x(x), y(y) { }

  inline bool operator< (const Point &other) const { 
    if (y < other.y)
      return true;
    if (y > other.y)
      return false;
    return x < other.x;
  }
};

typedef struct Point Point;

namespace hull {
namespace alg {
namespace cpu {

#ifdef BOOST_RANDOM
  typedef boost::mt19937 RNGType;

  static RNGType rng( time( 0 ) ); // produces randomness out of thin air
  static boost::uniform_real<> uniform_0_2pi( 0, 2 * M_PI );
  static boost::uniform_real<> uniform_0_1( 0, 1 );
  static boost::uniform_real<> uniform_m1_1( -1, 1 );
  static boost::variate_generator< RNGType, boost::uniform_real<> > random_angle( rng, uniform_0_2pi );
  static boost::variate_generator< RNGType, boost::uniform_real<> > random_radius( rng, uniform_0_1 );
  static boost::variate_generator< RNGType, boost::uniform_real<> > random_coord( rng, uniform_m1_1 );
#endif

  Point random_point();
  
  void draw_point( const Point & p, GLfloat size );
  void draw_point( const std::vector< Point > & vp, GLfloat size );
  void draw_point( const std::vector< thrust::tuple< float, float > > & vp, GLfloat size );

  void calculateConvexHull( vector< int > n_points );
  void calculateConvexHull( int n_points );

  vector<Point> convex_hull(vector<Point> P);
  thrust::host_vector<Point> convex_hull(thrust::host_vector<Point> & P);
  vector< thrust::tuple< float, float > > convex_hull(thrust::host_vector< thrust::tuple< float, float > > & P);

}
}
}
