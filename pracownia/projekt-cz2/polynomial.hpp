#ifndef __POLYNOMIAL_HPP
#define __POLYNOMIAL_HPP


template < typename dom = float, int N = 18 >
struct Polynomial
{
  // n is a degree of a polynomial, which means coeff array have n+1 elements
  dom coeff[N+1];
  dom coeff_der[N+1];

  int max_deg;

  __host__ __device__
  Polynomial( dom coeff_p[N+1] )
  {
    max_deg = 0;
    for(int i = 0; i < N+1; i++)
      {
        coeff[i] = 0;
        coeff_der[i] = 0;
        if (fabs (coeff_p[i]) > 0.00001)
          max_deg = i;
      }
    
    for(int i = 0; i < max_deg+1; i++)
      {
        coeff[max_deg-i] = coeff_p[i];
      }

    for(int i = 0; i < max_deg; i++)
      {
        coeff_der[i] = (max_deg-i) * coeff[i];
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


//__constant__ float arb_poly_const_coeff[3*18];
//__constant__ float arb_poly_const_coeff_der[3*18];
//__constant__ int arb_poly_const_size[3];


template < typename dom = float >
struct PolynomialSimple
{
  const int offset;

  __host__ __device__ PolynomialSimple(int offset) : offset(offset) { }

  __device__
  inline
  dom evaluate( const dom & x )
  {
    dom res = 0;
    int size = arb_poly_const_size[offset];
    int base = offset*(18+1);

    for(int i = 0; i < size+1; i++)
      {
        res *= x;
        res += arb_poly_const_coeff[base+i];
      }
    return res;
  }

  __device__
  inline
  dom derivative(const dom & x)
  {
    dom res = 0;
    int size = arb_poly_const_size[offset];
    int base = offset*(18+1);

    for(int i = 0; i < size; i++)
      {
        res *= x;
        res += arb_poly_const_coeff_der[base+i];
      }
    return res;
  }
};


#endif

