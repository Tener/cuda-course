
#include <vector_types.h>
#include <stdio.h>
#include "common.hpp"
#include "utils.hpp"


const std::string SurfString( Surf s )
{
  switch ( s )
    {
    case SURF_CHMUTOV:	 return std::string("SURF_CHMUTOV");
    case SURF_PLANE:	 return std::string("SURF_PLANE");
    case SURF_TORUS:	 return std::string("SURF_TORUS");
    case SURF_DING_DONG: return std::string("SURF_DING_DONG");
    case SURF_CAYLEY:	 return std::string("SURF_CAYLEY");
    case SURF_DIAMOND:	 return std::string("SURF_DIAMOND");
    case SURF_BALL:	 return std::string("SURF_BALL");      
    }
  return std::string("???");
}

void PrintVector( const float3 & Vec, char * name, std::ostream & out )
{
  char buf[4097];
  sprintf(buf, "%s=(%f,%f,%f)\n", name, Vec.x, Vec.y, Vec.z );
  out << std::string(buf);
}

void PrintView( View v, std::ostream & out)
{
  out << "steps=" << v.steps << std::endl;
  out << "surf=" << v.surf << "=" << SurfString(v.surf) << std::endl;
  PrintVector( v.StartingPoint, "start", out ); // what point is the center of our view?
  PrintVector( v.DirectionVector, "dirVec", out ); // in which direction and how far does it reach?
  out << "range_w=" << v.range_w << std::endl;
  out << "range_h=" << v.range_h << std::endl;

}
