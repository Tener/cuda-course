
#include "common.hpp"
#include "hull.hpp"
#include "graphics.hpp"

int main(int argn, char ** argv)
{
  if ( argn < 3 )
    {
      std::cerr << "Za mało argumentów. Oczekiwano minimum 2, było: " << argn-1 << endl;
      return 1;
    }

  Processor proc;

  if ( string("cpu") == string(argv[1]) )
    {
      proc = CPU;
    }
  else if ( string("gpu") == string(argv[1]) )
    {
      proc = GPU;
    }
  else
    {
      std::cerr << "Nie podano poprawnego procesora: " << string(argv[1]) << endl;
      return 1;
    }

  vector< int > num_points;
  for(int i = 2; i < argn; i++)
    {
      int np;
      sscanf(argv[i], "%d", &np);
      if( np > 0 )
	num_points.push_back( np );
    }

  hull::graphics::initGlWindow();

  for(vector< int >::iterator it = num_points.begin(); it < num_points.end(); it++)
    {
      hull::alg::calculateConvexHull( proc, *it );
    }

  hull::graphics::closeGlWindow();

  return 0;

}

