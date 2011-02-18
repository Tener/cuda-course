#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>
#include <cutil_math.h>
#include <curand_kernel.h>

#include <math_functions.h>

// thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

//png++
#include <png++/png.hpp>

// local includes
#define CHMUTOV_DEGREE 4

#include "constant_vars.hpp"

#include "common.hpp"
#include "graphics.hpp"

#include "polynomial.hpp"
#include "utils.hpp"
#include "chebyshev.hpp"

#include "colors.hpp"
#include "sign_change.hpp"
#include "linear_algebra.hpp"

#include "surface.hpp"

    
// Nice intro to ray tracing:
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm

template < typename dom3 = float3, typename dom = float >
struct ModelViewRay
{
  
  dom3 current_point;
  const dom3 direction_vector;
  dom position;

  __host__ __device__
  inline
  dom rescale(const dom dim, const dom dim_max, const dom scale) { return scale * ((dim - (dim_max/2))/dim_max); }

  __device__ __host__
  inline
  dom3 modelview_matrix_transform( const dom3 vec )
  {
    dom arrIn[4] = { vec.x, vec.y, vec.z, 1 }; // homo
    dom arrOut[4];

    for(int i = 0; i < 4; i++)
      {
        dom v = 0;
        for(int j = 0; j < 4; j++)
          {
#if __CUDA_ARCH__ > 0
            v += device_modelview_matrix[4*j + i] * arrIn[j];
#else
            v += host_modelview_matrix[4*j + i] * arrIn[j];
#endif
          }
        arrOut[i] = v;
      }

    return make_float3( arrOut[0] * arrOut[3], 
                        arrOut[1] * arrOut[3], 
                        arrOut[2] * arrOut[3] );
  } 

  __device__ __host__
  ModelViewRay( int w, int h, int ix_w, int ix_h, dom scale ) :
    current_point( modelview_matrix_transform( make_float3(rescale(ix_w,w,scale), rescale(ix_h,h,scale), 0) ) ),
    direction_vector( Normalize( modelview_matrix_transform( make_float3(0, 0, 1) ) ) ),
    position(0)
  {
  }

  // transform the current point along the ray by given distance. 
  // negative distance reverses direction of movement.
  __device__ __host__
  inline
  void move_point( const dom & length )
  {
    current_point.x += direction_vector.x * length;
    current_point.y += direction_vector.y * length;
    current_point.z += direction_vector.z * length;
    
    position += length;
  }

};

template < typename SurfType, typename RayType >
struct RayTrace
{
  uint background;
  SurfType surface;

  int steps;
  int bisect_count;

  int w; int h; int ix_h;
  float3 view_angle; float3 starting_point; float scale;
  float view_distance; // how far do we look

  int frame_cnt;

  RayTrace(int frame_cnt,
	   int w, int h, int ix_h,
           float3 view_angle, float3 starting_point, float scale,
           float view_distance, 
           int steps,
           int bisect_count,
           SurfType surface = SurfType())
  :
    frame_cnt(frame_cnt),
    w(w), h(h), ix_h(ix_h),
    view_angle(view_angle), starting_point(starting_point), scale(scale),
    view_distance(view_distance),
    steps(steps), bisect_count(bisect_count),
    background(0)
  {
  }

  __device__
  Color operator()( int ix_w )
  {
    RayType ray( w, h, ix_w, ix_h, 1 );
    float surf_value = surface.calculate( ray.current_point );

    Color4 color = make_float4( 0, 0, 0, 0 );

    const int MAX_ROOT = 1;

    for(int root_number = 0; root_number < MAX_ROOT; root_number++)
      {
        bool sign_change = false;
        float step = view_distance / steps;
        // root detection 
        for(; ray.position < view_distance && !sign_change;)
          {
	    step += (view_distance / steps); // if there is no root we go faster each step
            ray.move_point(step);
            float tmp = surface.calculate( ray.current_point );
            sign_change = SignChange<>::check( surf_value, tmp );
            surf_value = tmp;
          }
       
        if ( sign_change )
          {
            // root refinement
            for(int i = 0; i < bisect_count; i++)
              {
                /// XXX: check alternate implementation of step changing
                step /= 2;
                if ( sign_change )
                  {
                    step *= -1; // we reverse movement direction if there was a sign change
                  }
                
                ray.move_point(step);
                float tmp = surface.calculate( ray.current_point );
                sign_change = SignChange<>::check( surf_value, tmp );
                surf_value = tmp;
              }
            
            // shade calculation
	    surface.frame_cnt = frame_cnt;
            Color4 newcolor = surface.lightning(ray.current_point, 
						make_float3(1, 
							    sinf(frame_cnt * 0.02), 
							    cosf(frame_cnt * 0.03)));
            color = lerp( color, newcolor, expf( -root_number ));
          }
      }
    return make_color(color);
  }
};

template < typename SurfType, typename RayType = ModelViewRay< > >
struct RayTraceScreen
{
  int w;
  int h;
  View view;
  thrust::device_ptr< uint > dev_pbo;
    
  RayTraceScreen(int w, int h, View view, uint * pbo) :
    w(w), h(h), view(view), dev_pbo(pbo)
  {
  }

  static
  __host__
  png::rgba_pixel
  unpack_Color(const uint rgba)
  {
    char * rgba_arr = (char *)(&rgba);
    return png::rgba_pixel( rgba_arr[0], rgba_arr[1], rgba_arr[2], 255 );
  }

  struct PNGWriterArgs { 
    thrust::host_vector< uint > pixels;
    std::string filename;
    png::image< png::rgba_pixel > img;

    void write()
    {
      thrust::host_vector< uint >::iterator pix(pixels.begin());
        
      for(int i = 0; i < img.get_width(); i++)
        for(int j = 0; j < img.get_height(); j++)
          {
            img[j][i] = unpack_Color(*pix);
            pix++;
          }

      printf("writing file... %s\n", filename.c_str());
      img.write(filename);
      printf("file written! %s\n", filename.c_str());
    }

    static void * write_wrapper(void * arg)
    {
      PNGWriterArgs * pngwr = (PNGWriterArgs *) arg;
      pngwr->write();
      delete pngwr;
      return NULL;
    }    
  };

  __host__
  void screenshot(const std::string filename, int frame_cnt)
  {
    const int shot_width = 1600;
    const int shot_height = 1600;

    static thrust::device_vector< uint > dev_vector(shot_height * shot_width);  // made static to avoid frequent allocations
    render(&(dev_vector[0]), shot_width, shot_height, frame_cnt);
    
    PNGWriterArgs * args = new PNGWriterArgs;
    args->pixels = thrust::host_vector< uint >(dev_vector);
    args->filename = filename;
    args->img = png::image< png::rgba_pixel >(shot_width,shot_height);
    
    pthread_t th;
    pthread_create( &th, NULL, PNGWriterArgs::write_wrapper, (void*)args );
    pthread_join( th, NULL );
  }

  void render(thrust::device_ptr< uint > surface, int max_width, int max_height, int frame_count)
  {
    
    for(int ix_h = 0; ix_h < max_height; ix_h++)
    {
      thrust::transform( thrust::make_counting_iterator< short >(0),
                         thrust::make_counting_iterator< short >(max_width),
                         surface + max_height * ix_h,
                         RayTrace< SurfType, RayType >(frame_count,
						       max_width,max_height,ix_h,
                                                       view.angle, view.starting_point,
                                                       view.scale, view.distance, 
                                                       view.steps, view.bisect_count));
    }

  }

  void run()
  {
    static int frame_cnt = 0;
    frame_cnt++;
    
    render(dev_pbo, w, h, frame_cnt);

    if ( view.screenshot )
      {
        char filename[256];
        static int session = 0;
        static int cnt = 0;
        if (!session)
          {
            session = time(0);
            char cmd[1024];
            sprintf(cmd, "mkdir -p 'screenshots/%d/'",session);
            system(cmd);
          }

        sprintf(filename, "screenshots/%d/shot_%010d.png", session, cnt); // XXX: make '/' portable
        screenshot(std::string(filename), frame_cnt);
        
        // we made a screenshot. now, to not confuse 'movie' making part below, disable screenshot flag.
        cnt++;
        view.screenshot = false;
      }

    if ( view.movie )
    {
      static int session_start = 0;
      static int count = 0;
      static char path[1024];
      static View last_view;

      if (!session_start)
        {
          session_start = time(0);
          sprintf(path, "movie/%s/%d", SurfString(SurfType::surface_id).c_str(), session_start);
          // call 'mkdir -p' for recursive mkdir
          {
            char cmd[1024];
            sprintf(cmd, "mkdir -p '%s'", path);
            system(cmd);
          }
          last_view = view;
        }
      
      
      if (view.allframes || memcmp( (&last_view), (&view), sizeof(View)))
        {
          char filename[1024];
          sprintf(filename,"%s/%010d.png", path, count);
          count++;
          screenshot(std::string(filename), frame_cnt);
          last_view = view;
        }
    }

  }
};


void initModelViewMatrix(View view)
{
  PrintView( view );

  GLfloat modelViewMatrix[16];
  
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  {
    glLoadIdentity();
    glOrtho( -1, 1, -1, 1, -1, 1 );

    glScalef( view.scale, view.scale, view.scale ); // scale along z axis

    glRotatef( view.angle.x * 10, 1, 0, 0 );
    glRotatef( view.angle.y * 10, 0, 1, 0 );
    glRotatef( view.angle.z * 10, 0, 0, 1 );

    glTranslatef( view.starting_point.x/view.scale, 
                  view.starting_point.y/view.scale, 
                  view.starting_point.z/view.scale );

    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix);
  }
  glPopMatrix();

  for(int i = 0; i < 4; i++)
    {
      printf("\t");
      for(int j = 0; j < 4; j++)
        {
          printf("%f\t", modelViewMatrix[i+j*4]);
          host_modelview_matrix[i+j*4] = modelViewMatrix[i+j*4];
        }
      printf("\n");
    }

  cudaMemcpyToSymbol(device_modelview_matrix, host_modelview_matrix, sizeof(float) * 16);
}


extern "C" void launch_raytrace_kernel(uint * pbo, View view, int w, int h)
{
  // modelview matrix fun
  initModelViewMatrix(view);
  


#define TraceEngine RayTraceScreen

  switch ( view.surf )
    {
    case SURF_BARTH:
      TraceEngine< Surface< SURF_BARTH > >(w,h,view,pbo).run();
      break;
    case SURF_CHMUTOV:
      TraceEngine< Surface< SURF_CHMUTOV > >(w,h,view,pbo).run();
      break;
    case SURF_CHMUTOV_ALT:
      TraceEngine< Surface< SURF_CHMUTOV_ALT > >(w,h,view,pbo).run();
      break;
    case SURF_HEART:
      TraceEngine< Surface< SURF_HEART > >(w,h,view,pbo).run();
      break;
    case SURF_PLANE:
      TraceEngine< Surface< SURF_PLANE > >(w,h,view,pbo).run();
      break;
    case SURF_TORUS:
      TraceEngine< Surface< SURF_TORUS > >(w,h,view,pbo).run();
      break;
    case SURF_DING_DONG:
      TraceEngine< Surface< SURF_DING_DONG > >(w,h,view,pbo).run();
      break;
    case SURF_CAYLEY:
      TraceEngine< Surface< SURF_CAYLEY > >(w,h,view,pbo).run();
      break;
    case SURF_DIAMOND:
      TraceEngine< Surface< SURF_DIAMOND > >(w,h,view,pbo).run();
      break;
    case SURF_BALL:
      TraceEngine< Surface< SURF_BALL > >(w,h,view,pbo).run();
      break;
    case SURF_ARB_POLY:
      {
        // i'm tired of doing this the clean way... so let's just make a hack.
        // copy arbitrary polynomial's parameters
        for(int i = 0; i < 3; i++)
          {
            size_t stride = sizeof(float) * (18+1);
            Polynomial<> p(view.arb_poly[i]);
            cudaMemcpyToSymbol( arb_poly_const_coeff, p.coeff, stride, stride * i );
            cudaMemcpyToSymbol( arb_poly_const_coeff_der, p.coeff_der, stride, stride * i );
            stride = sizeof(int);
            cudaMemcpyToSymbol( arb_poly_const_size, &p.max_deg, stride, stride * i );
          }
        TraceEngine< Surface< SURF_ARB_POLY > >(w,h,view,pbo).run();
        break;
      }
    default:
      break;
    }

#undef TraceEngine
}
