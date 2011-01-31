#ifndef __POLYNOMIAL_HPP
#define __POLYNOMIAL_HPP

template < typename dom = float, int N = 18 >
struct Polynomial
{
  // n is a degree of a polynomial, which means coeff array have n+1 elements
  dom coeff[N+1];
  dom coeff_der[N];

  __host__ __device__
  Polynomial( dom coeff_p[N+1] )
  {
    for(int i = 0; i < N+1; i++)
      {
        coeff[i] = coeff_p[i];
        if ( i )
          {
            coeff_der[i-1] = coeff_p[i] * i;
          }
      }
  }

  // default constructor
  __host__ __device__
  Polynomial()
  {
  }

  __host__ __device__
  dom evaluate( dom x )
  {
    dom res = 0;
#pragma unroll 18
    for(int i = 0; i < N+1; i++)
      {
        res *= x;
        res += coeff[N+1-i];
      }
    return res;
  }

  __host__ __device__
  dom derivative(dom x)
  {
    dom res = 0;
#pragma unroll 18
    for(int i = 0; i < N; i++)
      {
        res *= x;
        res += coeff_der[N-i];
      }
    return res;
  }
};

#endif
