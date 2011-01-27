#include "surf.h"

struct View 
{
  Surf surf;
  int surf_i;
  float3 StartingPoint; // what point is the center of our view?
  float3 DirectionVector; // in which direction and how far does it reach?
  int steps;
  // bounding box elements:
  float range_w;
  float range_h;

View( Surf s = SURF_DING_DONG, 
      float3 start = make_float3( 0, 0, -100 ), 
      float3 dirvec = make_float3(0,0,1), 
      int steps = 100,
      float range_w = 50,
      float range_h = 50
      )
: surf(s), StartingPoint(start), DirectionVector(dirvec), steps(steps), range_w(range_w), range_h(range_h)
  { };
};

typedef struct View View;
