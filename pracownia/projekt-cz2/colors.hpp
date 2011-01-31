__host__ __device__
inline uint RGBA( unsigned char r, unsigned char g, unsigned char b, unsigned char a )
{ 
  return 
    (a << (3 * 8)) + 
    (b << (2 * 8)) +
    (g << (1 * 8)) +
    (r << (0 * 8));
}
