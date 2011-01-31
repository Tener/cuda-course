#include "surf.h"
#include "polynomial.hpp"







struct View 
{
  Surf surf;
  float3 StartingPoint; // what point is the center of our view?
  float3 DirectionVector; // in which direction and how far does it reach?
  int steps;
  int bisect_count;

  typedef Polynomial< float, 18 > PolyT;
  PolyT arb_poly[3];

  View(Surf s = SURF_CHMUTOV_1,
       float3 start = make_float3( 2.2, 1.9, 1.7 ), 
       float3 dirvec = make_float3(7.1,5.7,4.9), 
       int steps = 500,
       int bisect_count = 5)
  : surf(s), 
    StartingPoint(start), 
    DirectionVector(dirvec), 
    steps(steps)
  { 
    //  float chebyshev_coeff_16[16+1] = { +1, 0, -128, 0, +2688, 0, -21504, 0, +84480,  0, -180224, 0,  +212992, 0,  -131072, 0,  +32768};

    float chebyshev_coeff_18[18+1] = { -1, 0, +162, 0, -4320, 0, +44352, 0, -228096, 0, +658944, 0, -1118208, 0, +1105920, 0, -589824, 0, 131072 };

    PolyT chebyshev_Poly( chebyshev_coeff_18 );

    for(int i = 0; i < 3; i++)
      arb_poly[i] = chebyshev_Poly;
  };
};

typedef struct View View;
