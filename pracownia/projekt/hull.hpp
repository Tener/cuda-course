
#include "common.hpp"

namespace hull {
namespace alg {

  void calculateConvexHull( Processor proc, int points );

  namespace cpu {
    void calculateConvexHull( int points );
  }

  namespace gpu {
    void calculateConvexHull( int points );
  }

} // namespace alg
} // namespace hull
