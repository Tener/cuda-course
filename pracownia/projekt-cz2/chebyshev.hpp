__host__ __device__
inline
float Chebyshev( char n, float x )
{ // http://en.wikipedia.org/wiki/Chebyshev_polynomials
  return 
    ( x <= -1 ) ? ((n & 1 ? -1 : 1) * cosh( n * acosh( -x ) )) :
    (( x >= 1 ) ? cosh( n * acosh( x ) ) : cos(n * acos(x)));
};


__device__ __host__
float Chebyshev_Pol_N( int N, float x )
{
  float arr[CHMUTOV_DEGREE+1];
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
#pragma unroll 200
    for(unsigned int i = 0; i < (N/2); i++)
      {
        arr_0 = 2 * x * arr_1 - arr_0;
        arr_1 = 2 * x * arr_0 - arr_1;
      }
    return arr_0;
  }
};

template <int N>
struct Chebyshev_U_DiVar
{
  __host__ __device__
  static float calculate(float x)
  {
    float arr_0 = 1;
    float arr_1 = 2*x;
#pragma unroll 200
    for(unsigned int i = 0; i < (N/2); i++)
      {
        arr_0 = 2 * x * arr_1 - arr_0;
        arr_1 = 2 * x * arr_0 - arr_1;
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

template <int N>
struct Chebyshev_U
{
  __host__ __device__
  static float calculate(float x)
  { 
    return 2 * x * Chebyshev_U< N-1 >::calculate(x) - Chebyshev_U< N-2 >::calculate(x);
  };
};

template <>
struct Chebyshev_U< 0 >
{
  __host__ __device__
  static float calculate(float x)
  { 
    return 1;
  };
};

template <>
struct Chebyshev_U< 1 >
{
  __host__ __device__
  static float calculate(float x)
  { 
    return 2*x;
  };
};
