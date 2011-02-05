
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include <algorithm>

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand_kernel.h>


namespace rt {
  namespace graphics {

    class GLManager
    {
    public:
      int height;
      int width;

      void initGlWindow();
      void closeGlWindow();
      void reshape(int w, int h);

      GLManager( int w = 1024, int h = 1024 ) : width(w), height(h) { };
        ~GLManager() { };
    };

    extern GLManager global_glm;

    struct VBO { 
      // vbo variables
      GLuint vbo;
      struct cudaGraphicsResource *cuda_vbo_resource;
      float4 * dev_vbo;
      unsigned int n_points;
      unsigned int draw_points;

      VBO(unsigned int n_points, unsigned int vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard);
      ~VBO();

      void map();
      void unmap();
      void render( int point_cnt = -1, float3 color = make_float3( 0.3, 0.65, 0.23 ) );

    };

    struct PBO {
      GLuint pbo;
      uint * dev_pbo; // device ptr to PBO

      int width;
      int height;

      PBO(int width, int height);
      ~PBO() { };
      void map();
      void unmap();
      void render();
    };

    // much like scoped ptr
    template < typename Resource >
    class ScopedMapping {
    
    private:
      Resource & res;
    
    public:
      ScopedMapping( Resource & res ) : res(res) {
        res.map();
      }

      ~ScopedMapping() {
        res.unmap();
      }

    };



  } // namespace graphics
} // namespace rt

