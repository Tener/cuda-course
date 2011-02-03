#include "colors.hpp"



template <Surf surface, typename Vector = float3, typename dom = float>
struct Surface {
  __host__ __device__
  Surface() { };

  __host__ __device__ 
  dom calculate(const Vector & V);

  __host__ __device__ 
  dom calculate(const dom & x, const dom & y, const dom & z){ return calculate( make_float3( x, y, z ) ); }
    
  __host__ __device__
  Color lightning(Vector V, Vector Light)
  {
    return RGBA( COLOR_EXPDAMP( V.x * 2 ),
                 COLOR_EXPDAMP( V.y * 2 ),
                 COLOR_EXPDAMP( V.z * 2 ),
                 0);
  }
};

template < typename Vector, typename dom >
struct Surface< SURF_CHMUTOV, Vector, dom >
{
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
  Color lightning(Vector V, Vector Light)
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

    

    return RGBA( 30 + 100 + 100 * dot_pr_r,
                 30 + 100 + 100 * dot_pr_g,
                 30 + 100 + 100 * dot_pr_b,
                 0);
  }
};

template <>
__host__ __device__
float Surface< SURF_CHMUTOV_ALT >::calculate(const float3 & V)
{
    return Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate( V.x ) + 
           Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate( V.y ) + 
           Chebyshev_DiVar< CHMUTOV_DEGREE >::calculate( V.z );
}


template < typename Vector, typename dom >
struct Surface< SURF_ARB_POLY, Vector, dom >
{
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
  Color lightning(Vector V, Vector Light)
  {          
    float dot_pr = 0;
    dot_pr = DotProduct( Light.x, Light.y, Light.z,
			 params_s_x.derivative(V.x),
			 params_s_y.derivative(V.y),
			 params_s_z.derivative(V.z));
    
    return RGBA( 30 + 100 + 100 * dot_pr,
		 0, 0, 0 );        

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
float Surface< SURF_HEART >::calculate(const float3 & V)
{
    float x = V.x; float y = V.y; float z = V.z;
    return pow(2*x*x+y*y+z*z-1,3) - (0.1*x*x+y*y)*z*z*z;
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
