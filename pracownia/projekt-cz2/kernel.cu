#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

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



__host__ __device__
float DotProduct(float a_x, float a_y, float a_z,
                 float b_x, float b_y, float b_z)
{
  return 
    (a_x * b_x +
     a_y * b_y +
     a_z * b_z)
    / (VecMagnitude( a_x, a_y, a_z ) * VecMagnitude( b_x, b_y, b_z ));
}

__host__ __device__
void Normalize( float3 & Vec )
{
  float len = sqrt(Vec.x * Vec.x + Vec.y * Vec.y + Vec.z * Vec.z);
  Vec.x /= len;
  Vec.y /= len;
  Vec.z /= len;
}

    
// Nice intro to ray tracing:
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm

template < typename SurfTyp >
struct TracePoint
{
  const uint background;

  int w; int h; int ix_h;
  Surf surf;
  int steps;
  int bisect_count;
  
  float3 R0;
  float3 Rd;
  
  float3 Rtrans;

  SurfTyp surfaceInstance;

  //  typedef Polynomial< float, 16 > Poly;
  //Poly poly_surf[3];

  // bounding box
  float range_w;
  float range_h;

  // Vmin, Vmax, Vdiff;
  float3 Vmin, Vmax, Vdiff;
  float step_size;
  
  TracePoint(int w, int h, 
             int ix_h, 
             View v,
             SurfTyp surfInst = SurfTyp())
    : w(w), h(h), ix_h(ix_h), 
      surf(v.surf),
      steps(v.steps),
      bisect_count(v.bisect_count),
      R0(v.StartingPoint),
      Rd(v.DirectionVector),
      step_size(sqrt(pow(R0.x - Rd.x,2) + 
		     pow(R0.y - Rd.y,2) + 
		     pow(R0.z - Rd.z,2)) / steps),
      Rtrans(make_float3( R0.x - Rd.x, R0.y - Rd.y, R0.z - Rd.z )),
      background(0),
      surfaceInstance(surfInst)
  { 
    range_w = sqrt(pow(R0.x - Rd.x,2) + 
                   pow(R0.y - Rd.y,2) + 
                   pow(R0.z - Rd.z,2)) / 2;
    range_h = range_w;

    Vmin.x = R0.x - range_w;
    Vmin.y = R0.y - range_h;
    Vmin.z = R0.z;

    Vmax.x = Rd.x + range_w;
    Vmax.y = Rd.y + range_h;
    Vmax.z = Rd.z;

    Vdiff.x = Vmax.x - Vmin.x;
    Vdiff.y = Vmax.y - Vmin.y;
    Vdiff.z = Vmax.z - Vmin.z;

    // x, y, z
    /*
      1, 0, -128, 0, +2688, 0, -21504, 0, +84480, 0, -180224, 0, +212992, 0, -131072, 0, +32768, 0
     
    */

//    float chebyshev_coeff_18[18+1] = { -1, 0, +162, 0, -4320, 0, +44352, 0, -228096, 0, +658944, 0, -1118208, 0, +1105920, 0, -589824, 0, 131072 };
//    float chebyshev_coeff_16[16+1] = { +1, 0, -128, 0, +2688, 0, -21504, 0, +84480,  0, -180224, 0,  +212992, 0,  -131072, 0,  +32768};
// 
//    poly_surf[0] = Poly( chebyshev_coeff_18 );
//    poly_surf[1] = Poly( chebyshev_coeff_18 );
//    poly_surf[2] = Poly( chebyshev_coeff_18 );

    //    surfaceInstance = SurfaceT();
  };

  __host__ __device__
  inline
  void Ray( float3 & Rc, const float3 & Rd, const float & t )
  {
    Rc.x += Rd.x * t;
    Rc.y += Rd.y * t;
    Rc.z += Rd.z * t;
  };


  __host__ __device__ 
  uint operator()( int ix_w )
  {
    float x = 2.0f * (float)range_w * (((float)ix_w - (w/2.0f)) / (float)w);
    float y = 2.0f * (float)range_h * (((float)ix_h - (h/2.0f)) / (float)w);
    float z = 0 ;

    float3 Rc = make_float3( R0.x + x, R0.y + y, R0.z + z ); // current point
    
    float val = surfaceInstance.calculate( Rc );
    bool sign_has_changed = false;
    
    for(int i = 0; (i < steps) && !sign_has_changed; i++)
      {
 	// calculate next position
        Ray( Rc, Rtrans, step_size );
 	// determine the sign
 	float tmp = surfaceInstance.calculate(Rc);
	sign_has_changed = SignChange<>::check( val, tmp );
 	val = tmp;
      }

      if ( sign_has_changed )
     {
       float step_size_l = step_size;
#pragma unroll
       for(int i = 0; i < bisect_count && i < 11; i++)
	 {
	   step_size_l /= 2 * (1 + (sign_has_changed * -2 ));
// 	   if ( sign_has_changed )
// 	     {
// 	       step_size *= -1; // if there was a sign change, we swap directions
// 	     }
	   //
	   Ray( Rc, Rtrans, step_size_l );
	   float tmp = surfaceInstance.calculate(Rc);
	   sign_has_changed = SignChange<>::check( val, tmp ); //SignChange( val, tmp );
	   val = tmp;
	 }
       return surfaceInstance.lightning(Rc, make_float3( 1, 0, 0 ));

     }
      else
        {
          return background;
        }
  }
};

template < typename SurfTyp >
struct TraceScreen
{
  static
  void run(int w, int h, View view, uint * pbo, SurfTyp s = SurfTyp())
  {
    for(int ix_h = 0; ix_h < h; ix_h++)
    {
      thrust::transform( thrust::make_counting_iterator< short >(0),
                         thrust::make_counting_iterator< short >(w),
                         thrust::device_ptr< uint >(pbo + h * ix_h),
                         TracePoint< SurfTyp >(w,h,ix_h,view,s));
    }
  }
};


extern "C" void launch_raytrace_kernel(uint * pbo, View view, int w, int h)
{
  std::cerr << "w=" << w << std::endl
            << "h=" << h << std::endl; 

  PrintView( view );

  // i'm tired of doing this the clean way... so let's just make a hack.
  {
    for(int i = 0; i < 3; i++)
      {
        size_t stride = sizeof(float) * (18+1);
        Polynomial<> p(view.arb_poly[i]);
        cudaMemcpyToSymbol( arb_poly_const_coeff, p.coeff, stride, stride * i );
        cudaMemcpyToSymbol( arb_poly_const_coeff_der, p.coeff_der, stride, stride * i );

 //       std::cout << "BUUUU " << i << " " << p.max_deg <<  "\n";
 //       for(int ii = 0; ii < (18+1); ii++)
 //         {
 //           std::cout << 
 //             "   " << ii << 
 //             "\t" << p.coeff_der[ii] << 
 //             "\t" << p.coeff[ii] << 
 //             "\t" << view.arb_poly[i][ii] << 
 //             "\n";
 //         }

        stride = sizeof(int);
        cudaMemcpyToSymbol( arb_poly_const_size, &p.max_deg, stride, stride * i );
      }

  }

  switch ( view.surf )
    {
    case SURF_CHMUTOV_0:
    case SURF_CHMUTOV_1:
    case SURF_CHMUTOV_2:
    case SURF_CHMUTOV_3:
      TraceScreen< Surface< SURF_CHMUTOV_1 > >::run(w,h,view,pbo);
      break;
    case SURF_HEART:
      TraceScreen< Surface< SURF_HEART > >::run(w,h,view,pbo);
      break;
    case SURF_PLANE:
      TraceScreen< Surface< SURF_PLANE > >::run(w,h,view,pbo);
      break;
    case SURF_TORUS:
      TraceScreen< Surface< SURF_TORUS > >::run(w,h,view,pbo);
      break;
    case SURF_DING_DONG:
      TraceScreen< Surface< SURF_DING_DONG > >::run(w,h,view,pbo);
      break;
    case SURF_CAYLEY:
      TraceScreen< Surface< SURF_CAYLEY > >::run(w,h,view,pbo);
      break;
    case SURF_DIAMOND:
      TraceScreen< Surface< SURF_DIAMOND > >::run(w,h,view,pbo);
      break;
    case SURF_BALL:
      TraceScreen< Surface< SURF_BALL > >::run(w,h,view,pbo);
      break;
    case SURF_ARB_POLY:
      TraceScreen< Surface< SURF_ARB_POLY > >::run(w,h,view,pbo);
        //                                                   Surface< SURF_ARB_POLY, float3, float >(view.arb_poly));
      break;
    default:
      break;
    }
  
  //for(int ix_h = 0; ix_h < h; ix_h++)
  //  {
  //    thrust::transform( thrust::make_counting_iterator< short >(0),
  //                       thrust::make_counting_iterator< short >(w),
  //                       thrust::device_ptr< uint >(pbo + h * ix_h),
  //                       Surface< SURF_CHMUTOV_1, float3, float, void > >(w,h,ix_h,view));
  //  }

}
