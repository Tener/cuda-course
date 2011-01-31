
typedef uint Color;



template <Surf surface, typename Vector = float3, typename dom = float>
struct Surface {
  __host__ __device__
  Surface() { };

  __host__ __device__ 
  dom calculate(const Vector & V){ return 0; };
    
  __host__ __device__
  Color lightning(Vector V, Vector Light)
  {
#define EXPDAMP( p ) (10.0f + 240.0f * (expf(-fabsf(p))))
    return RGBA( EXPDAMP( V.x ),
                 EXPDAMP( V.y ),
                 EXPDAMP( V.z ),
                 0);
#undef EXPDAMP
  }

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

};

template < typename Vector, typename dom >
struct Surface< SURF_CHMUTOV_1, Vector, dom >
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
  Color lightning(Vector V, Vector Light)
  {
    float dot_pr = DotProduct( Light.x, Light.y, Light.z,
			       Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.x ),
			       Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.y ),
			       Chebyshev_U< CHMUTOV_DEGREE >::calculate( V.z ));

    return RGBA( 30 + 100 + 100 * dot_pr,
		 0, 0, 0 );
  }
};

template < typename Vector, typename dom >
struct Surface< SURF_ARB_POLY, Vector, dom >
{
  typedef Polynomial< dom, 18 > PolyT;
  PolyT params[3];
 
  __host__ __device__
  Surface(PolyT p[3]){ params[0] = p[0]; params[1] = p[1]; params[2] = p[2];};

  __host__ __device__
  Surface() : params( )
  { };


  __host__ __device__ 
  inline
  dom calculate(const Vector & V)
  {
    // XXX: fixme
    return params[0].evaluate(V.x) + params[1].evaluate(V.y) + params[2].evaluate(V.z);
    
  }

  __host__ __device__
  Color lightning(Vector V, Vector Light)
  {          
    float dot_pr = 0;
//    dot_pr = DotProduct( Light.x, Light.y, Light.z,
//        		 this->params[0].derivative(V.x),
//        		 this->params[1].derivative(V.y),
//        		 this->params[2].derivative(V.z));
    
    return RGBA( 30 + 100 + 100 * dot_pr,
		 0, 0, 0 );        

  }
};

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


