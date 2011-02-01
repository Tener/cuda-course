#ifndef __COLORS_HPP
#define __COLORS_HPP

typedef uint Color;

__host__ __device__
inline uint RGBA( unsigned char r, unsigned char g, unsigned char b, unsigned char a )
{ 
  return 
    (a << (3 * 8)) + 
    (b << (2 * 8)) +
    (g << (1 * 8)) +
    (r << (0 * 8));
}

#define COLOR_EXPDAMP( p ) (10.0f + 240.0f * (expf(-fabsf(p))))
#define COLOR_BYRANGE( p, pmin, pmax ) (10.0f + 240.0f * fabs((p-pmin)/(pmax-pmin)) )
#define COLOR_UNIT_RANGE( x ) fabs(240 * (x + 1) / 2)


//#define COLOR( p, pmin, pmax ) (10.0f + 240.0f * fabs((p-pmin)/(pmax-pmin)) )
// 
//       return RGBA( COLOR( Rc.x, Vmin.x, Vmax.x ),
//        	    COLOR( Rc.y, Vmin.y, Vmax.y ),
//        	    COLOR( Rc.z, Vmin.z, Vmax.z ),
//        	    0);
// 
//#undef COLOR
 

//#define TRANS( x ) fabs(240 * (x + 1) / 2)
// 
//       return RGBA( TRANS(Rc.x) + 10, 
// 		    TRANS(Rc.y) + 10,
// 		    TRANS(Rc.z) + 10,
//                    0); 
//#undef TRANS


#endif
