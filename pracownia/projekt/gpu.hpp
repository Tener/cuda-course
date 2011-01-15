
#include "common.hpp"

namespace hull {
  namespace alg {
    namespace gpu {

      void calculateConvexHull( vector< int > n_points );
      void calculateConvexHull( int n_points );

      void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
		     unsigned int vbo_res_flags);

      void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);



    }
  }
}
