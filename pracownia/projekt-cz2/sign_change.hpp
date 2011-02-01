template < typename dom = float >
struct SignChange {

  __host__ __device__
  inline
  static 
  // default implementation
  bool check( const dom & a, const dom & b )
  {
    return checkSlow( a, b );
  }  

  __host__ __device__
  inline
  static 
  // this is likely to be slow
  bool checkSlow( const dom & a, const dom & b )
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
  }

  __host__ __device__
  inline
  static 
  bool checkBit( const float & a, const float & b )
  {
    return signbit(a) != signbit(b);
  }

  __host__ __device__
  inline
  static 
  bool checkFast( const float & a, const float & b )
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
};
