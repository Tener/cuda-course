#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/times.h>

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

#include <thrust/generate.h>
#include <thrust/host_vector.h>

inline bool SignChangeSlow( const float & a, const float & b )
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


// this is likely to be slow
//bool SignChangeSlow( const float & a, const float & b )
inline bool SignChangeSlow_t( const thrust::tuple< float, float > & t )
{
  float a = thrust::get<0>(t);
  float b = thrust::get<1>(t);
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

inline bool Less( const thrust::tuple< float, float > & t )
{
  float a = thrust::get<0>(t);
  float b = thrust::get<1>(t);

  return a < b;
}

inline bool SignChange( const float & a, const float & b )
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
  bool d2 = d0 ^ d1; // false if a == 0
  bool d3 = b < 0;
  bool d4 = b > 0;
  bool d5 = d3 ^ d4; // false if b == 0
    
  return (d0 ^ d3) || (d1 ^ d4) || (d2 ^ d5);
}


inline bool SignChange_t( const thrust::tuple< float, float > & t )
{
  float a = thrust::get<0>(t);
  float b = thrust::get<1>(t);
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
  bool d2 = d0 ^ d1; // false if a == 0
  bool d3 = b < 0;
  bool d4 = b > 0;
  bool d5 = d3 ^ d4; // false if b == 0
    
  return (d0 ^ d3) || (d1 ^ d4) || (d2 ^ d5);
}

float drand48_shifted()
{
  return drand48() - 0.5;
}


void testRandom(int n = 1024 * 1024)
{
  struct tms buf;

  srand48(time(0));

  const int SIZE = 1024 * 1024 * 512;

  std::cerr << "Generate randoms" << std::endl;

  thrust::host_vector< float > rands( SIZE );
  clock_t now = times(&buf); clock_t tmp;
  //thrust::generate( rands.begin(), rands.end(), drand48_shifted );
  thrust::sequence( rands.begin(), rands.end(), -1 * (rands.size() / 2) );
  
  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;
  {
    bool changed = false;
    for(int i = 0; i < SIZE-1; i++)
      {
	changed |= SignChange_t( thrust::make_tuple( rands[i], rands[i+1] ) );
      }
    std::cerr << changed << std::endl;
  }

  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;
  {
    bool changed = false;
    for(int i = 0; i < SIZE-1; i++)
      {
	changed |= SignChangeSlow_t( thrust::make_tuple( rands[i], rands[i+1] ) );
      }
    std::cerr << changed << std::endl;
  }

  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;
  std::cerr << thrust::transform_reduce( thrust::make_zip_iterator( thrust::make_tuple( rands.begin(), rands.begin()+1 ) ),
					 thrust::make_zip_iterator( thrust::make_tuple( rands.end()-1, rands.end() ) ),
					 Less, 0, thrust::maximum<float>() ) << std::endl;

  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;
  std::cerr << thrust::transform_reduce( thrust::make_zip_iterator( thrust::make_tuple( rands.begin(), rands.begin()+1 ) ),
					 thrust::make_zip_iterator( thrust::make_tuple( rands.end()-1, rands.end() ) ),
					 SignChange_t, 0, thrust::maximum<bool>() ) << std::endl;

  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;
  std::cerr << thrust::transform_reduce( thrust::make_zip_iterator( thrust::make_tuple( rands.begin(), rands.begin()+1 ) ),
					 thrust::make_zip_iterator( thrust::make_tuple( rands.end()-1, rands.end() ) ),
					 SignChangeSlow_t, 0, thrust::maximum<bool>() ) << std::endl;

  tmp = times(&buf); std::cerr << tmp - now << std::endl; now = tmp;

  
}

void test( float a, float b)
{
  std::cout << "(a= " << a << ", b=" << b << ") " 
	    << SignChange( a, b ) << " " 
	    << SignChangeSlow( a, b ) << std::endl;
 
}

int main()
{
#if 0
  test(-1, -1);
  test(-1,  0);
  test(-1,  1);

  test(0, -1);
  test(0,  0);
  test(0,  1);

  test(1, -1);
  test(1,  0);
  test(1,  1);  

  test(-5, -5);
  test(-5,  0);
  test(-5,  5);

  test(0, -5);
  test(0,  0);
  test(0,  5);

  test(5, -5);
  test(5,  0);
  test(5,  5);  
#endif
  testRandom();

  return 0;
}
