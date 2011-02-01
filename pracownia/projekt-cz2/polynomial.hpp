#ifndef __POLYNOMIAL_HPP
#define __POLYNOMIAL_HPP

template < typename dom = float, int N = 18 >
struct Polynomial
{
  // n is a degree of a polynomial, which means coeff array have n+1 elements
  dom coeff[N+1];
  dom coeff_der[N];

  int max_deg;

  __host__ __device__
  Polynomial( dom coeff_p[N+1] )
  {
    
    for(int i = 0; i < N+1; i++)
      {
        dom c = coeff_p[i];
        coeff[N-i] = c;
        if ( c != 0 )
          max_deg = i;
      }

    for(int i = 0; i < N; i++)
      {
        coeff_der[i] = (N-i) * coeff[i];
      }

  }
  __host__ __device__
  Polynomial()
  {
  }
  
//  __host__ __device__
//  Polynomial()
//  {
//    printf("Default const.\n");
//    for(int i = 0; i < N+1; i++)
//      {
//        coeff[i] = i;
//      }
// 
//    for(int i = 0; i < N; i++)
//      {
//        coeff_der[i] = i*(i+1);
//      }
//  }
 // __host__ __device__
 // inline
 // dom evalDer( const dom & x, dom & y, dom & yd )
 // {
 //   y = 0; yd = 0;
 //   for(int i = 0; i < N; i++)
 //     {
 //       dom c = coeff[i];
 //       y *= x;
 //       y += c;
 //       yd *= x;
 //       yd += (N-1) * c;
 //     }
 //   y *= x;
 //   y += coeff[N];
 // }

  __host__ __device__
  inline
  dom evaluate( const dom & x )
  {
    dom res = 0;
    int iter_cnt = MAX( max_deg, N )+1;
    for(int i = 0; i < iter_cnt; i++)
      {
        res *= x;
        res += coeff[i];
      }
    return res;
  }

  __host__ __device__
  inline
  dom derivative(const dom & x)
  {
    dom res = 0;
    int iter_cnt = MAX( max_deg, N );
    for(int i = 0; i < iter_cnt; i++)
      {
        res *= x;
        res += coeff_der[i];
      }
    return res;
  }
};

#endif
