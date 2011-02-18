#include "surf.h"
//#include "polynomial.hpp"

struct View 
{
  Surf surf;
  float3 starting_point; // what point is the center of our view?
  float3 DirectionVector; // in which direction and how far does it reach?
  int steps;
  int bisect_count;

  float3 angle;
  float scale;
  float distance;

  bool screenshot;
  bool movie;
  bool asyncRender;
  bool allframes;

  float arb_poly[3][18+1];

  View(Surf s = SURF_CHMUTOV,
       float3 start = make_float3(-0.1,0.1,-0.9),
       float3 dirvec = make_float3(7.1,5.7,4.9), 
       int steps = 500,
       int bisect_count = 15)
  : surf(s), 
    starting_point(start), 
    DirectionVector(dirvec), 
    steps(steps),
    scale(2.5),
    distance(1.6),
    bisect_count(bisect_count),
    angle(make_float3(-1.4, 6.8, -9.4)),
    screenshot(false),
    movie(false),
    asyncRender(true),
    allframes(true)
  { 
    float chebyshev_coeff_18[18+1] = { -1, 0, +162, 0, -4320, 0, +44352, 0, -228096, 0, +658944, 0, -1118208, 0, +1105920, 0, -589824, 0, 131072 };

    for(int i = 0; i < 3; i++)
      {
        for(int j = 0; j < 18+1; j++)
          {
            arb_poly[i][j] = chebyshev_coeff_18[j];
          }
      }

  };
};

typedef struct View View;
