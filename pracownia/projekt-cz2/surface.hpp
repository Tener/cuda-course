#include "colors.hpp"

__host__ __device__
inline
float scale_dot_pr( float dot_pr )
{
    const float p = 0.3;
    return (dot_pr * p) + (1-p);
}

template <Surf surface, typename Vector = float3, typename dom = float>
struct Surface {
  static const Surf surface_id = surface;
  int frame_cnt;

  __host__ __device__
  Surface() { };

  __host__ __device__ 
  dom calculate(const Vector & V);

  __host__ __device__ 
  dom calculate(const dom & x, const dom & y, const dom & z){ return calculate( make_float3( x, y, z ) ); }
    
  __host__ __device__
  Color4 lightning(Vector V, Vector Light)
  {
    return make_float4( COLOR_EXPDAMP( V.x * 2 ),
                        COLOR_EXPDAMP( V.y * 2 ),
                        COLOR_EXPDAMP( V.z * 2 ),
                        0);
  }
};

template < typename Vector, typename dom >
struct Surface< SURF_CHMUTOV, Vector, dom >
{
  static const Surf surface_id = SURF_CHMUTOV;
  int frame_cnt;

  __host__ __device__ 
  inline
  dom calculate(const Vector & V)
  {
    return Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.x ) + 
           Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.y ) + 
           Chebyshev_T< CHMUTOV_DEGREE >::calculate( V.z );
  }

  __host__ __device__ 
  inline
  dom calculate(const dom & x, const dom & y, const dom & z)
  {
    return Chebyshev_T< CHMUTOV_DEGREE >::calculate( x ) + 
           Chebyshev_T< CHMUTOV_DEGREE >::calculate( y ) + 
           Chebyshev_T< CHMUTOV_DEGREE >::calculate( z );
  }

  __host__ __device__
  Color4 lightning(Vector V, Vector Light)
  {
    float dot_pr_r = DotProduct( Light.x, Light.y, Light.z,
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.x ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.y ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.z ));

    float dot_pr_g = DotProduct( Light.x+1, Light.y+1, Light.z+1,
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.x ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.y ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.z ));

    float dot_pr_b = DotProduct( Light.x+2, Light.y-2, Light.z-3,
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.x ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.y ),
                                 Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.z ));

    return make_float4( scale_dot_pr( dot_pr_r ),
                        scale_dot_pr( dot_pr_g ),
                        scale_dot_pr( dot_pr_b ),
                        0 );

  }
};

#define CHMUTOV_DI_DEGREE 4

template <>
__host__ __device__
float Surface< SURF_CHMUTOV_ALT >::calculate(const float3 & V)
{
    return Chebyshev_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.x ) + 
           Chebyshev_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.y ) + 
           Chebyshev_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.z );
}

__host__ __device__
float CalcLight(float x1, float y1, float z1,
		float x2, float y2, float z2)
{
  return DotProduct( x1 / VecMagnitude(x1,y1,z1),
		     y1 / VecMagnitude(x1,y1,z1),
		     z1 / VecMagnitude(x1,y1,z1),
		     x2 / VecMagnitude(x2,y2,z2),
		     y2 / VecMagnitude(x2,y2,z2),
		     z2 / VecMagnitude(x2,y2,z2));
}

template <>
__host__ __device__
Color4 Surface< SURF_CHMUTOV_ALT >::lightning(float3 V, float3 Light)
{
  float dx = Chebyshev_U_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.x );
  float dy = Chebyshev_U_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.y );
  float dz = Chebyshev_U_DiVar< CHMUTOV_DI_DEGREE >::calculate( V.z );

  float dot_pr_r = CalcLight( Light.x, Light.y, Light.z,
			      dx, dy, dz );
  
  float dot_pr_g = CalcLight( Light.x+1, Light.y+1, Light.z+1,
			      dx, dy, dz );
  
  float dot_pr_b = CalcLight( Light.x+2, Light.y-2, Light.z-3,
			      dx, dy, dz );
  
  return make_float4( scale_dot_pr( dot_pr_r),
		      scale_dot_pr( dot_pr_g),
		      scale_dot_pr( dot_pr_b),
		      0 );
}



template < typename Vector, typename dom >
struct Surface< SURF_ARB_POLY, Vector, dom >
{
  static const Surf surface_id = SURF_ARB_POLY;
  int frame_cnt;

  PolynomialSimple<> params_s_x;
  PolynomialSimple<> params_s_y;
  PolynomialSimple<> params_s_z;

  __host__ __device__
  Surface() :
    params_s_x(0),
    params_s_y(1),
    params_s_z(2)
  {
  }

  __device__ 
  inline
  dom calculate(const Vector & V)
  {
    return params_s_x.evaluate(V.x) + params_s_y.evaluate(V.y) + params_s_z.evaluate(V.z);    
  }

  __device__ 
  dom calculate(const dom & x, const dom & y, const dom & z){ return calculate( make_float3( x, y, z ) ); }

  __device__
  Color4 lightning(Vector V, Vector Light)
  {          
    float dot_pr = 0;
    dot_pr = DotProduct( Light.x, Light.y, Light.z,
			 params_s_x.derivative(V.x),
			 params_s_y.derivative(V.y),
			 params_s_z.derivative(V.z));
    
    return make_float4( scale_dot_pr( dot_pr ),
                        0,
                        0,
                        0 );

  }
};

template <>
__host__ __device__
float Surface< SURF_BARTH >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;
    float phi = 1.618033988; //(1 + sqrtf(5))/2;
    float phi4 = phi * phi * phi * phi;

    return (5*phi+3)*((x*x)+(y*y)+(z*z)-1)*((x*x)+(y*y)+(z*z)-1)*((x*x)+(y*y)+(z*z)+phi-2)*((x*x)+(y*y)+(z*z)+phi-2)+
      8*((x*x*x*x)-2*(x*x)*(y*y)-2*(x*x)*(z*z)+(y*y*y*y)-2*(y*y)*(z*z)+(z*z*z*z))*((x*x)-(y*y)*phi4)*((z*z)-(x*x)*phi4)*((y*y)-(z*z)*phi4);

}

template <>
__host__ __device__
Color4 Surface< SURF_HEART >::lightning(float3 V, float3 Light)
{
    float x = V.x; float y = V.y; float z = V.z;
    // derivatives
    // dx: 12 x (2 x^2+y^2+z^2-1)^2-0.2 x z^3
    // dy: 6 y (2 x^2+y^2+z^2-1)^2-2 y z^3
    // dz: z (6 (2 x^2+y^2+z^2-1)^2-3. z (0.1 x^2+y^2))

    float dx=12*x*pow(2*x*x+y*y+z*z-1,2)-0.2*x*z*z*z;
    float dy=6*y*pow(2*x*x+y*y+z*z-1,2)-2*y*z*z*z;
    float dz=z*(6*pow(2*x*x+y*y+z*z-1,2)-3.0*z*(0.1*x*x+y*y));

    float dot_pr_r = CalcLight( Light.x+0.5*sin(x+y+z),
				Light.y+0.3*sin(x+y+z), 
				Light.z+0.2*sin(x+y+z),
				dx, dy, dz );
    
    float dot_pr_g = CalcLight( Light.x+1*sin(x+y+z),
				Light.y-1.1*sin(x+y+z),
				Light.z+1.3*sin(x+y+z),
				dx, dy, dz );
    
    float dot_pr_b = CalcLight( Light.x+2*sin(x+y+z), 
				Light.y-2*sin(x+y+z),
				Light.z-3*sin(x+y+z),
				dx, dy, dz );
    
    return make_float4( scale_dot_pr(dot_pr_r),
			scale_dot_pr(dot_pr_g),
			scale_dot_pr(dot_pr_b),
			0 );
}


template <>
__host__ __device__
float Surface< SURF_HEART >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;
    return pow(2*x*x+y*y+z*z-1,3) - (0.1*x*x+y*y)*z*z*z;

    // derivative: 12 x (2 x^2+y^2+z^2-1)^2-0.2 x z^3
}

template <>
__host__ __device__
float Surface< SURF_PLANE >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;
    float A = 1, B = 2, C = 3, D = 2.1;

    return A*x + B*y + C*z + D;
}

template <>
__host__ __device__
float Surface< SURF_TORUS >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;

    float R = 1;
    float r = .3;
    return pow(R - sqrt(x*x + y*y), 2 ) + z*z - r*r;

}

template <>
__host__ __device__
float Surface< SURF_DING_DONG >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;

    return x*x+y*y-z*(1-z*z);
}

template <>
__host__ __device__
float Surface< SURF_DIAMOND >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;

    return sin(x) * sin(y) * sin(z) 
         + sin(x) * cos(y) * cos(z) 
         + cos(x) * sin(y) * cos(z) 
         + cos(x) * cos(y) * sin(z);
}

template <>
__host__ __device__
float Surface< SURF_BALL >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;

    return sqrt(x * x + y * y + z * z) - 1;
}

template <>
__host__ __device__
float Surface< SURF_CAYLEY >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;

    return -5 * (x * x * (y + z) + y * y * (x + z) + z * z * (x + y)) + 2 * (x * y + y * x + x * z);
}
