#include "surf.h"

struct View 
{
  Surf surf;
  int surf_i;
  float3 StartingPoint; // what point is the center of our view?
  float3 DirectionVector; // in which direction and how far does it reach?
  int steps;
  int bisect_count;
  // bounding box elements:
  float range_w;
  float range_h;

View( Surf s = SURF_CHMUTOV_1,
      float3 start = make_float3( 2.2, 1.9, 1.7 ), 
      float3 dirvec = make_float3(7.1,5.7,4.9), 
      int steps = 500,
      float range_w = 8,
      float range_h = 8,
      int bisect_count = 5
      )
: surf(s), StartingPoint(start), DirectionVector(dirvec), steps(steps), range_w(range_w), range_h(range_h)
  { };
};

typedef struct View View;
