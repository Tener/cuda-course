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

#include "common.hpp"
#include "graphics.hpp"

#include "utils.hpp"

#define CHMUTOV_DEGREE 16

__host__ __device__
inline uint RGBA( unsigned char r, unsigned char g, unsigned char b, unsigned char a )
{ 
  return 
    (a << (3 * 8)) + 
    (b << (2 * 8)) +
    (g << (1 * 8)) +
    (r << (0 * 8));
}

__device__ __host__
float Chebyshev_Pol_N( int N, float x )
{
  float arr[CHMUTOV_DEGREE+1];
  //  thrust::device_vector< float > arr( N );
  arr[0] = 1;
  arr[1] = x;
#pragma unroll 16
  for(unsigned int i = 2; i < N+1; i++)
    {
      arr[i] = 2 * x * arr[i-1] - arr[i-2];
    }
  return arr[N];
}

template <int N>
struct Chebyshev_DiVar
{
  __host__ __device__
  static float calculate(float x)
  {
    float arr_0 = 1;
    float arr_1 = x;
#pragma unroll 16
    for(unsigned int i = 1; i < N/2; i++)
      {
        arr_0 = 2 * x * arr_0 - arr_1;
        arr_1 = 2 * x * arr_1 - arr_0;
      }
    return arr_0;
  }
};


template <int N>
struct Chebyshev_Pol
{
  __host__ __device__
  static float calculate(float x)
  {
    float arr[N+1];
    arr[0] = 1;
    arr[1] = x;
#pragma unroll 16
    for(unsigned int i = 2; i < N+1; i++)
      {
	arr[i] = 2 * x * arr[i-1] - arr[i-2];
      }
    return arr[N];
  }
};

template <int N>
struct Chebyshev_T
{
  __host__ __device__
  static float calculate(float x)
  { 
    return 2 * x * Chebyshev_T< N-1 >::calculate(x) - Chebyshev_T< N-2 >::calculate(x);
  };
};

template <>
struct Chebyshev_T< 0 >
{
  __host__ __device__
  static float calculate(float x)
  { 
    return 1;
  };
};

template <>
struct Chebyshev_T< 1 >
{
  __host__ __device__
  static float calculate(float x)
  { 
    return x;
  };
};


//inline 
//__host__ __device__ 
//float Chebyshev_Pol( int N, float x )
//{
//  
//  if ( N == 0 )
//    return 1;
//  if ( N == 1 )
//    return x;
//  return 2 * x * Chebyshev_Pol(N-1, x) - Chebyshev_Pol(N-2, x);
//}


    
// Nice intro to ray tracing:
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm

struct TracePoint
{
  int w; int h; int ix_h;
  Surf surf;
  int steps;
  int bisect_count;
  
  float3 R0;
  float3 Rd;
  
  float3 Rtrans;

  // bounding box
  float range_w;
  float range_h;

  // Vmin, Vmax, Vdiff;
  float3 Vmin, Vmax, Vdiff;

  float step_size;
  
  TracePoint(int w, int h, 
             int ix_h, 
             View v)
    : w(w), h(h), ix_h(ix_h), 
      surf(v.surf),
      steps(v.steps),
      bisect_count(v.bisect_count),
      R0(v.StartingPoint),
      Rd(v.DirectionVector),
      range_w(v.range_w),
      range_h(v.range_h),
      step_size(sqrt(pow(R0.x - Rd.x,2) + 
		     pow(R0.y - Rd.y,2) + 
		     pow(R0.z - Rd.z,2)) / steps),
      Rtrans(make_float3( R0.x - Rd.x, R0.y - Rd.y, R0.z - Rd.z ))
  { 
    Vmin.x = R0.x - range_w;
    Vmin.y = R0.y - range_h;
    Vmin.z = R0.z;

    Vmax.x = Rd.x + range_w;
    Vmax.y = Rd.y + range_h;
    Vmax.z = Rd.z;

    Vdiff.x = Vmax.x - Vmin.x;
    Vdiff.y = Vmax.y - Vmin.y;
    Vdiff.z = Vmax.z - Vmin.z;
  };

  __host__ __device__
  inline
  float Chebyshev( char n, float x )
  { // http://en.wikipedia.org/wiki/Chebyshev_polynomials
    return 
      ( x <= -1 ) ? ((n & 1 ? -1 : 1) * cosh( n * acosh( -x ) )) :
      (( x >= 1 ) ? cosh( n * acosh( x ) ) : cos(n * acos(x)));
  };

  __host__ __device__
  inline
  float Surface(float3 V,enum Surf surf_id)
  {
    float x, y, z;
    x = V.x; y = V.y; z = V.z;

    switch ( surf_id )
      {
      case SURF_DING_DONG:
        {
          return x*x+y*y-z*(1-z*z);
        }

      case SURF_CHMUTOV_0:
	return Chebyshev( CHMUTOV_DEGREE, V.x ) + 
	       Chebyshev( CHMUTOV_DEGREE, V.y ) + 
	       Chebyshev( CHMUTOV_DEGREE, V.z );

      case SURF_CHMUTOV_1:
	return Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.x ) + 
	       Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.y ) + 
	       Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.z );

      case SURF_CHMUTOV_2:
	return Chebyshev_Pol< CHMUTOV_DEGREE >::calculate(V.x)
             + Chebyshev_Pol< CHMUTOV_DEGREE >::calculate(V.y)
             + Chebyshev_Pol< CHMUTOV_DEGREE >::calculate(V.z);

      case SURF_CHMUTOV_3:
	return Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate(V.x)
	     + Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate(V.y)
	     + Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate(V.z);

//        return Chebyshev_Pol_N( CHMUTOV_DEGREE, V.x)
//             + Chebyshev_Pol_N( CHMUTOV_DEGREE, V.y)
//             + Chebyshev_Pol_N( CHMUTOV_DEGREE, V.z);

      case SURF_TORUS:
        {
          float c = .3;
          float a = 3;
          return pow(c - x*x + y*y, 2 ) + z*z - a*a;
        }
      case SURF_DIAMOND:
        {
          return sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) + cos(x) * cos(y) * sin(z);
        }
      case SURF_BALL:
        {
          return sqrt(x * x + y * y + z * z) - 1;
        }
      case SURF_CAYLEY:
        {
          return -5 * (x * x * (y + z) + y * y * (x + z) + z * z * (x + y)) + 2 * (x * y + y * x + x * z);
        }
      }

    return 0;
        
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
  inline
  // this is likely to be slow
  bool SignChangeSlow( const float & a, const float & b )
  {
    if ( a < 0 ) // a is below 0
      {
	return !(b < 0);
      }
    else 
      if (a > 0) // a is above 0
	{
	  return !(b > 0);
	}
      else // a is equal to 0
	{
	  return (b != 0);
	}
  };

  __host__ __device__
  bool SignChangeBit( const float & a, const float & b )
  {
    return signbit(a) != signbit(b);
  }

  __host__ __device__
  bool SignChange( const float & a, const float & b )
  {
    /*
0      a < 0
1      a > 0
2      0 ^ 1

3      b < 0
4      b > 0
5      3 ^ 4

     (0 ^ 3)
  || (1 ^ 4)
  || (2 ^ 5)
  
    */
    
    bool d0 = a < 0;
    bool d1 = a > 0;
    bool d2 = d0 ^ d1;
    bool d3 = b < 0;
    bool d4 = b > 0;
    bool d5 = d3 ^ d4;
    
    return (d0 ^ d3) || (d1 ^ d4) || (d2 ^ d5);
  }

  __host__ __device__
  void Normalize( float3 & Vec )
  {
    float len = sqrt(Vec.x * Vec.x + Vec.y * Vec.y + Vec.z * Vec.z);
    Vec.x /= len;
    Vec.y /= len;
    Vec.z /= len;
  }

  __host__ __device__ 
  uint operator()( int ix_w )
  {
    float x = 2.0f * (float)range_w * (((float)ix_w - (w/2.0f)) / (float)w);
    float y = 2.0f * (float)range_h * (((float)ix_h - (h/2.0f)) / (float)w);
    float z = 0 ;

    float3 Rc = make_float3( R0.x + x, R0.y + y, R0.z + z ); // current point
    
    float val = Surface( Rc, surf );
    bool sign_has_changed = false;
    
    for(int i = 0; (i < steps) && !sign_has_changed; i++)
      {
 	// calculate next position
        Ray( Rc, Rtrans, step_size );
 	// determine the sign
 	float tmp = Surface(Rc,surf);
	sign_has_changed = SignChangeSlow( val, tmp ); //SignChange( val, tmp );
 	val = tmp;
      }

      if ( sign_has_changed )
     {
       float step_size_l = step_size;
#pragma unroll 10
       for(int i = 0; i < bisect_count; i++)
	 {
	   step_size_l /= sign_has_changed ? -2 : 2;
// 	   if ( sign_has_changed )
// 	     {
// 	       step_size *= -1; // if there was a sign change, we swap directions
// 	     }
	   //
	   Ray( Rc, Rtrans, step_size_l );
	   float tmp = Surface(Rc,surf);
	   sign_has_changed = SignChangeSlow( val, tmp ); //SignChange( val, tmp );
	   val = tmp;
	 }

#define EXPDAMP( p ) (10.0f + 240.0f * (expf(-fabsf(p))))
       return RGBA( EXPDAMP( Rc.x ),
                    EXPDAMP( Rc.y ),
                    EXPDAMP( Rc.z ),
                    0);


//#define COLOR( p, pmin, pmax ) (10.0f + 240.0f * fabs((p-pmin)/(pmax-pmin)) )
// 
//       return RGBA( COLOR( Rc.x, Vmin.x, Vmax.x ),
//        	    COLOR( Rc.y, Vmin.y, Vmax.y ),
//        	    COLOR( Rc.z, Vmin.z, Vmax.z ),
//        	    0);
// 
//#undef COLOR
 

//#define TRANS( x ) fabs(240 * (x + 1) / 2)
// 
//       return RGBA( TRANS(Rc.x) + 10, 
// 		    TRANS(Rc.y) + 10,
// 		    TRANS(Rc.z) + 10,
//                    0); 
//#undef TRANS
       
     }
      else
        {
          return RGBA( 0, 
                       0,
                       0,  
                       0);
        }
  }
};

extern "C" void launch_raytrace_kernel(uint * pbo, View view, int w, int h)
{
  std::cerr << "w=" << w << std::endl
            << "h=" << h << std::endl; 

  PrintView( view );

  for(int ix_h = 0; ix_h < h; ix_h++)
    {
      thrust::transform( thrust::make_counting_iterator< short >(0),
                         thrust::make_counting_iterator< short >(w),
                         thrust::device_ptr< uint >(pbo + h * ix_h),
                         TracePoint(w,h,ix_h,
                                    view));
    }

}
