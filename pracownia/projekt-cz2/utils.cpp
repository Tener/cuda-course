
#include <vector_types.h>
#include <stdio.h>
#include "common.hpp"
#include "utils.hpp"


const std::string SurfString( Surf s )
{
  switch ( s )
    {
    case SURF_BARTH:	 return std::string("SURF_BARTH");
    case SURF_CHMUTOV: return std::string("SURF_CHMUTOV");
    case SURF_CHMUTOV_ALT: return std::string("SURF_CHMUTOV_ALT");
    case SURF_PLANE:	 return std::string("SURF_PLANE");
    case SURF_HEART:	 return std::string("SURF_HEART");
    case SURF_TORUS:	 return std::string("SURF_TORUS");
    case SURF_DING_DONG: return std::string("SURF_DING_DONG");
    case SURF_CAYLEY:	 return std::string("SURF_CAYLEY");
    case SURF_DIAMOND:	 return std::string("SURF_DIAMOND");
    case SURF_BALL:	 return std::string("SURF_BALL");      
    case SURF_ARB_POLY:  return std::string("SURF_ARB_POLY");
    }
  return std::string("???");
}

void PrintVector( const float3 & Vec, const char * name, std::ostream & out )
{
  char buf[4097];
  sprintf(buf, "%s=(%f,%f,%f)\n", name, Vec.x, Vec.y, Vec.z );
  out << std::string(buf);
}

void PrintView( View v, std::ostream & out)
{
  out << "steps=" << v.steps << std::endl;
  out << "bisect_count=" << v.bisect_count << std::endl;
  out << "surf=" << v.surf << "=" << SurfString(v.surf) << std::endl;
  PrintVector( v.starting_point, std::string("start").c_str(), out );
  PrintVector( v.DirectionVector, std::string("dirVec").c_str(), out );
  PrintVector( v.angle, std::string("angle").c_str(), out );
  out << "scale=" << v.scale << std::endl;
  out << "distance=" << v.distance << std::endl;

}
