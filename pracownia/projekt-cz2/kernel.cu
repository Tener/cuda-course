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
    
    
    
// Nice intro to ray tracing:
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter0.htm

struct TracePoint
{
  int w; int h; int ix_h;

  float3 R0; // origin point

  TracePoint(int w, int h, int ix_h, float3 R0 = make_float3( -1, -1, -1 ) ) : w(w), h(h), ix_h(ix_h), R0(R0) { };

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
  float Surface( float3 V )
  {
    // for now - let's choose chebyshev's polynomials
    return Chebyshev( CHMUTOV_DEGREE, V.x ) + Chebyshev( CHMUTOV_DEGREE, V.y ) + Chebyshev( CHMUTOV_DEGREE, V.z );
  };

  __host__ __device__
  inline
  void Ray( float3 & Rc, const float3 & R0, const float3 & Rd, const float & t )
  {
    Rc.x = R0.x + Rd.x * t;
    Rc.y = R0.y + Rd.y * t;
    Rc.z = R0.z + Rd.z * t;
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
    bool d3 = b > 0;
    bool d4 = b < 0;
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
  void PrintVector( const float3 & Vec )
  {
    printf("Vec=(%f,%f,%f)\n", Vec.x, Vec.y, Vec.z );
  }

  __host__ __device__ 
  uint operator()( int ix_w )
  {
   const float max_range = 1;
   const float min_range = 0;
   const float max_cnt = 50;
   const float step = ((float)(max_range - min_range)) / max_cnt;
   
   // directon vector
   float3 Rd = make_float3( -(float)(w/2) + ix_w, 
                            -(float)(h/2) + ix_h, 
                            1); 
   // must be normalized!
   Normalize( Rd );
   PrintVector( Rd );

   float3 Rc; // current point
   
   float val = Surface( R0 ); // current surface value
   bool sign_has_changed = false;
   for(int i = 0; (i < max_cnt) && !sign_has_changed; i++)
      {
 	// calculate our current position
 	Ray( Rc, R0, Rd, i * step );
 	// determine the sign
 	float tmp = Surface( Rc );
	
	sign_has_changed = SignChangeSlow( val, tmp ); //SignChange( val, tmp );
 	val = tmp;
      }
 
   if ( sign_has_changed )
     {
       //       printf("(%f,%f,%f)\n", Rc.x, Rc.y, Rc.z);

#define TRANS( x ) (240 * (x + 1) / 2)

       //       printf("%f\n%f\n%f\n", Rc.x, Rc.y, Rc.z);

       return RGBA( TRANS(Rc.x) + 10, 
                    TRANS(Rc.y) + 10,
                    TRANS(Rc.z) + 10,
                    0); 

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


extern "C" void launch_raytrace_kernel(uint * pbo, int w, int h)
{
  std::cerr << "w=" << w << std::endl
            << "h=" << h << std::endl; 

  static float3 R = make_float3(0,0,-1);
  static float cnt = 0;

  R.x = 3 * sin(cnt);
  R.y = 3 * cos(cnt);
  
  cnt += .01;

  for(int ix_h = 0; ix_h < h; ix_h++)
    {
      thrust::transform( thrust::make_counting_iterator< short >(0),
                         thrust::make_counting_iterator< short >(w),
                         thrust::device_ptr< uint >(pbo + h * ix_h),
                         TracePoint(w,h,ix_h,R) );
    }

}
