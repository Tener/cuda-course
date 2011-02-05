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
#define CHMUTOV_DEGREE 16

#include "constant_vars.hpp"

#include "common.hpp"
#include "graphics.hpp"

#include "polynomial.hpp"
#include "utils.hpp"
#include "chebyshev.hpp"

#include "colors.hpp"
#include "sign_change.hpp"
#include "surface.hpp"

#include "linear_algebra.hpp"


    
// Nice intro to ray tracing:
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm

//// PerspectiveRay class; base_vector specifies how we should locate ourselves in the space
//template < typename dom3 = float3, typename dom = float >
//struct PerspectiveRay
//{
//  int w;
//  int h;
//  int ix_w;
//  int ix_h;
//  dom3 direction_vector;
//  dom3 view_angle;
//  dom3 starting_point;
// 
//  dom3 current_point;
//  
//  __host__ __device__
//  PerspectiveRay( int w, int h, int ix_w, int ix_h, dom3 view_angle, dom3 starting_point, dom scale, dom3 base_vector = make_float3(1,0,0) )
//    : w(w), h(h), ix_w(ix_w), ix_h(ix_h), view_angle(view_angle), starting_point(starting_point), 
//      // XXX: direction vector should depend on w/h/ix_w/ix_h 
//      direction_vector(rotate_vector< dom3, dom >(base_vector, view_angle)) 
//  {
//  }
// 
//  // transform the current point along the ray by given distance. 
//  // negative distance reverses direction of movement.
//  __host__ __device__
//  void move_point( const dom & length )
//  {
//    current_point.x += direction_vector.x * length;
//    current_point.y += direction_vector.y * length;
//    current_point.z += direction_vector.z * length;
//  }
//};
// 
//// OrtographicRay class; base_vector specifies how we should locate ourselves in the space
//template < typename dom3 = float3, typename dom = float >
//struct OrtographicRay
//{
//  int w;
//  int h;
//  int ix_w;
//  int ix_h;
//  dom3 direction_vector;
//  dom3 view_angle;
// 
//  dom3 current_point;
// 
//  __host__ __device__
//  inline
//  dom rescale(const dom dim, const dom dim_max, const dom scale) { return scale * ((dim - (dim_max/2))/dim_max); }
//  
//  __host__
//  OrtographicRay( int w, int h, int ix_w, int ix_h, dom3 view_angle, dom3 starting_point, dom scale, dom3 base_vector = make_float3(1,0,0) )
//    : w(w), h(h), ix_w(ix_w), ix_h(ix_h), view_angle(view_angle),
//      direction_vector(rotate_vector< dom3, dom >(base_vector, view_angle)),
//      // XXX: starting point should depend on w/h/ix_w/ix_h AND direction vector
//      current_point(translate_point(starting_point, make_float3(0, rescale(ix_w,w,scale), rescale(ix_h,h,scale))))
//      //      current_point(translate_point(starting_point, rotate_vector< dom3, dom >(make_float3(0, rescale(ix_w,w,scale), rescale(ix_h,h,scale)), view_angle)))
//  {
//  }
// 
//  // transform the current point along the ray by given distance. 
//  // negative distance reverses direction of movement.
//  __host__ __device__
//  void move_point( const dom & length )
//  {
//    current_point.x += direction_vector.x * length;
//    current_point.y += direction_vector.y * length;
//    current_point.z += direction_vector.z * length;
//  }
//};

template < typename dom3 = float3, typename dom = float >
struct ModelViewRay
{
  
  dom3 current_point;
  const dom3 direction_vector;

  __host__ __device__
  inline
  dom rescale(const dom dim, const dom dim_max, const dom scale) { return scale * ((dim - (dim_max/2))/dim_max); }

  __device__ __host__
  inline
  dom3 modelview_matrix_transform( const dom3 vec )
  {
    //device_modelview_matrix[16]

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
    direction_vector( Normalize( modelview_matrix_transform( make_float3(0, 0, 1) ) ) )
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

  RayTrace(int w, int h, int ix_h,
           float3 view_angle, float3 starting_point, float scale,
           float view_distance, 
           int steps,
           int bisect_count,
           SurfType surface = SurfType())
  :
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
//    RayType ray( w, h, ix_w, ix_h, // which pixel on screen are we calculating
//        	 view_angle,	   // where do we look
//        	 starting_point,   // where do we start
//        	 scale		   // defines the length of '1.0' in pixels
//        	 );

    RayType ray( w, h, ix_w, ix_h, 1 );

    float surf_value = surface.calculate( ray.current_point );
    bool sign_change = false;
    float step = view_distance / steps;
    float pos = 0; // position along the ray

    // root detection 
    for(; pos < view_distance && !sign_change;)
      {
        step += (view_distance / steps) / 10; // if there is no root we go faster each step
        ray.move_point(step);
        pos += step;
        float tmp = surface.calculate( ray.current_point );
        sign_change = SignChange<>::check( surf_value, tmp );
        surf_value = tmp;
      }



    if ( sign_change )
      {
        // root refinement
        for(int i = 0; i < bisect_count; i++)
	 {
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
        return surface.lightning(ray.current_point, make_float3(1, 0, 0));
      }
    else
      {
        return background;
      }  
  }
};

// 
//template < typename SurfType >
//struct TracePoint
//{
//  const uint background;
//  int steps;
//  int bisect_count;
//  SurfType surfaceInstance;
// 
//  // all these variables are related to the viewport:
//  // - this will likely stay:
//  int w; int h; int ix_h; 
//  float3 R0;
//  // - but not these:
//  float3 Rd;
//  float3 Rtrans;
//  float range_w; // bounding box
//  float range_h; // bounding box
// 
//  // Vmin, Vmax, Vdiff;
//  float3 Vmin, Vmax, Vdiff;
//  float step_size;
//  
//  TracePoint(int w, int h, 
//             int ix_h, 
//             View v,
//             SurfType surfInst = SurfType())
//    : w(w), h(h), ix_h(ix_h), 
//      steps(v.steps),
//      bisect_count(v.bisect_count),
//      R0(v.starting_point),
//      Rd(v.DirectionVector),
//      step_size(sqrt(pow(R0.x - Rd.x,2) + 
//        	     pow(R0.y - Rd.y,2) + 
//        	     pow(R0.z - Rd.z,2)) / steps),
//      Rtrans(make_float3( R0.x - Rd.x, R0.y - Rd.y, R0.z - Rd.z )),
//      background(0),
//      surfaceInstance(surfInst)
//  { 
//    range_w = sqrt(pow(R0.x - Rd.x,2) + 
//                   pow(R0.y - Rd.y,2) + 
//                   pow(R0.z - Rd.z,2)) / 2;
//    range_h = range_w;
// 
//    Vmin.x = R0.x - range_w;
//    Vmin.y = R0.y - range_h;
//    Vmin.z = R0.z;
// 
//    Vmax.x = Rd.x + range_w;
//    Vmax.y = Rd.y + range_h;
//    Vmax.z = Rd.z;
// 
//    Vdiff.x = Vmax.x - Vmin.x;
//    Vdiff.y = Vmax.y - Vmin.y;
//    Vdiff.z = Vmax.z - Vmin.z;
//  };
// 
//  __host__ __device__
//  inline
//  void Ray( float3 & Rc, const float3 & Rd, const float & t )
//  {
//    Rc.x += Rd.x * t;
//    Rc.y += Rd.y * t;
//    Rc.z += Rd.z * t;
//  };
// 
//  __host__ __device__ 
//  uint operator()( int ix_w )
//  {
//    float x = 2.0f * (float)range_w * (((float)ix_w - (w/2.0f)) / (float)w);
//    float y = 2.0f * (float)range_h * (((float)ix_h - (h/2.0f)) / (float)w);
//    float z = 0 ;
// 
//    float3 Rc = make_float3( R0.x + x, R0.y + y, R0.z + z ); // current point
//    
//    float val = surfaceInstance.calculate( Rc );
//    bool sign_has_changed = false;
//    
//    for(int i = 0; (i < steps) && !sign_has_changed; i++)
//      {
// 	// calculate next position
//        Ray( Rc, Rtrans, step_size );
// 	// determine the sign
// 	float tmp = surfaceInstance.calculate(Rc);
//        sign_has_changed = SignChange<>::check( val, tmp );
// 	val = tmp;
//      }
// 
//      if ( sign_has_changed )
//     {
//       float step_size_l = step_size;
//#pragma unroll
//       for(int i = 0; i < bisect_count && i < 11; i++)
//         {
//           step_size_l /= 2 * (1 + (sign_has_changed * -2 ));
//// 	   if ( sign_has_changed )
//// 	     {
//// 	       step_size *= -1; // if there was a sign change, we swap directions
//// 	     }
//           //
//           Ray( Rc, Rtrans, step_size_l );
//           float tmp = surfaceInstance.calculate(Rc);
//           sign_has_changed = SignChange<>::check( val, tmp ); //SignChange( val, tmp );
//           val = tmp;
//         }
//       return surfaceInstance.lightning(Rc, make_float3( 1, 0, 0 ));
// 
//     }
//      else
//        {
//          return background;
//        }
//  }
//};
// 
//template < typename SurfType >
//struct TraceScreen
//{
//  static
//  void run(int w, int h, View view, uint * pbo, SurfType s = SurfType())
//  {
//    for(int ix_h = 0; ix_h < h; ix_h++)
//    {
//      thrust::transform( thrust::make_counting_iterator< short >(0),
//                         thrust::make_counting_iterator< short >(w),
//                         thrust::device_ptr< uint >(pbo + h * ix_h),
//                         TracePoint< SurfType >(w,h,ix_h,view,s));
//    }
//  }
//};





template < typename SurfType, typename RayType = ModelViewRay< > >
struct RayTraceScreen
{
  int w;
  int h;
  View view;
  uint * pbo;

    
  RayTraceScreen(int w, int h, View view, uint * pbo) :
    w(w), h(h), view(view), pbo(pbo)
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

  __host__
  void screenshot(const std::string filename)
  {
    png::image< png::rgba_pixel > img(w,h);
    thrust::device_ptr< uint > dev_pbo(pbo);
    thrust::host_vector< uint > pixels(dev_pbo, dev_pbo + w * h );

    thrust::host_vector< uint >::iterator pix(pixels.begin());
        
    for(int i = 0; i < img.get_width(); i++)
      for(int j = 0; j < img.get_height(); j++)
        {
          img[i][j] = unpack_Color(*pix);
          pix++;
        }
    img.write(filename);
    printf("file written! %s\n", filename.c_str());
  }

  void run()
  {
    for(int ix_h = 0; ix_h < h; ix_h++)
    {
      thrust::transform( thrust::make_counting_iterator< short >(0),
                         thrust::make_counting_iterator< short >(w),
                         thrust::device_ptr< uint >(pbo + h * ix_h),
                         RayTrace< SurfType, RayType >(w,h,ix_h,
                                                       view.angle, view.starting_point,
                                                       view.scale, view.distance, 
                                                       view.steps, view.bisect_count));
    }


    if ( view.screenshot )
      {
        char filename[256];
        sprintf(filename, "screenshots/shot_%d.png", time(0)); // XXX: make '/' portable
        screenshot(std::string(filename));
        
        // we made the screenshot. now, to not confuse 'movie' making part below, disable screenshot flag.
        view.screenshot = false;
      }

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
            //mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
          }
          last_view = view;
        }
      
      if (memcmp( (&last_view), (&view), sizeof(View)))
        {
          char filename[1024];
          sprintf(filename,"%s/%010d.png", path, count);
          count++;
          screenshot(std::string(filename));
          last_view = view;
        }
    }

  }
};

//template < typename RayType >
//struct RayDebug
//{  
//  int steps;
//  int bisect_count;
// 
//  int w; int h; int ix_h;
//  float3 view_angle; float3 starting_point; float scale;
//  float view_distance; // how far do we look
// 
//  float4 * vbo;
// 
//  RayDebug(int w, int h, int ix_h,
//           float3 view_angle, float3 starting_point, float scale,
//           float view_distance, 
//           int steps,
//           int bisect_count,
//           float4 * vbo)
//  :
//    w(w), h(h), ix_h(ix_h),
//    view_angle(view_angle), starting_point(starting_point), scale(scale),
//    view_distance(view_distance),
//    steps(steps), bisect_count(bisect_count),
//    vbo(vbo)
//  {
//  }
// 
//  __device__ __host__
//  void operator()( int ix_w )
//  {
//    //    RayType ray( w, h, ix_w, ix_h, scale );
//    RayType ray( w, h, ix_w, ix_h, // which pixel on screen are we calculating
//                 view_angle,       // where do we look
//                 starting_point,   // where do we start
//                 scale             // defines the length of '1.0' in pixels
//                 );
// 
//    float step = view_distance / steps;
// 
//    vbo += steps * ix_w;
// 
//    for(int i = 0; i < steps; i++)
//      {
//        ray.move_point(step);
//        *vbo = make_float4( ray.current_point.x, ray.current_point.y, ray.current_point.z, 1.0 );
//        vbo++;
//      }
//  }
//};
// 
//template < typename RayType = ModelViewRay< > >
//struct DebugRayTraceScreen
//{
//  static
//  void run(int w, int h, View view, float4 * vbo, uint * draw_cnt)
//  {
//    view.steps = MIN(view.steps, MAX_DEBUG_STEPS);
// 
//    for(int ix_h = 0; ix_h < h; ix_h++)
//    {
//      thrust::for_each( thrust::make_counting_iterator< short >(0),
//                        thrust::make_counting_iterator< short >(w),
//                        RayDebug< OrtographicRay< > >(w,h,ix_h,
//                                            view.angle, view.starting_point,
//                                            view.scale, view.distance, 
//                                            view.steps, view.bisect_count,
//                                            vbo + view.steps * ix_h * w) );
//    }
//    *draw_cnt = view.steps * h * w;
//  }
//};

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

//extern "C" void launch_debug_kernel(float4 * vbo, unsigned int * draw_cnt, View view, int w, int h)
//{
//  *draw_cnt = 0;
//  initModelViewMatrix(view);
//  DebugRayTraceScreen< >::run( w, h, view, vbo, draw_cnt );
// 
//}


extern "C" void launch_raytrace_kernel(uint * pbo, View view, int w, int h)
{
//  std::cerr << "w=" << w << std::endl
//            << "h=" << h << std::endl; 

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
