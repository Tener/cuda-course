
#include "hull.hpp"

namespace hull {
namespace alg {

  void calculateConvexHull( Processor proc, int points )
  {
    switch( proc )
      {
      case CPU:
	hull::alg::cpu::calculateConvexHull( points );
	break;
      case GPU:
	hull::alg::gpu::calculateConvexHull( points );
	break;
      default:
	cerr << "Bad processor" << endl; 
	return ;
      }
  }

} // namespace alg
} // namespace hull
