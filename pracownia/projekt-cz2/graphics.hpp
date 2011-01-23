
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

      GLManager( int w = 512, int h = 512 ) : width(w), height(h) { };
        ~GLManager() { };
    };

    extern GLManager global_glm;

    struct VBO { 
      // vbo variables
      GLuint vbo;
      struct cudaGraphicsResource *cuda_vbo_resource;
      void * d_vbo_buffer;
      unsigned int n_points;

      VBO(unsigned int n_points, unsigned int vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard);

      ~VBO();

      void render( int point_cnt = -1, float3 color = make_float3( 0.3, 0.65, 0.23 ) );
      float4 * mapResourcesGetMappedPointer();
      void unmapResources();

    };

    struct PBO {
      GLuint pbo;
      uint *dev_pbo; // device ptr to PBO

      int width;
      int height;

      PBO(int width, int height);
      ~PBO() { };
      void mapBufferObject();
      void unmapBufferObject();
      void render();
    };

    // much like scoped ptr
    class PBO_map_unmap {
    
    private:
      PBO & pbo;
    
    public:
      PBO_map_unmap( PBO & pbo ) : pbo(pbo) {
        pbo.mapBufferObject();
      }

      ~PBO_map_unmap() {
        pbo.unmapBufferObject();
      }

    };



  } // namespace graphics
} // namespace rt

