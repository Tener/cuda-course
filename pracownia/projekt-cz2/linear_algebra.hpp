__host__ __device__
float VecMagnitude(float x, float y, float z)
{
  return sqrt(x*x+y*y+z*z);
}

__host__ __device__
float DotProduct(float a_x, float a_y, float a_z,
                 float b_x, float b_y, float b_z)
{
  return 
    (a_x * b_x +
     a_y * b_y +
     a_z * b_z)
    / (VecMagnitude( a_x, a_y, a_z ) * VecMagnitude( b_x, b_y, b_z ));
}

__host__ __device__
void Normalize( float3 & Vec )
{
  float len = sqrt(Vec.x * Vec.x + Vec.y * Vec.y + Vec.z * Vec.z);
  Vec.x /= len;
  Vec.y /= len;
  Vec.z /= len;
}

template < typename dom >
__host__ __device__
inline
void multiply_matrices(dom A[3][3], const dom B[3][3] )
{
  
}

template < typename dom3, typename dom >
__host__ __device__
inline
void multiply_vector_matrix(dom3 vec, const dom M[3][3])
{

}


template < typename dom3, typename dom >
__host__ __device__
inline
dom3 rotate_vector(const dom3 vec, const dom3 euler_angles)
{
  dom a, b, c; 
  a = euler_angles.x;
  b = euler_angles.y;
  c = euler_angles.z;

  const dom R_x[3][3] = { { cos(a), 0, sin(a) },
                          { 0, 1, 0 },
                          { - sin(a), 0, cos(a) } };

  const dom R_y[3][3] = { { 1, 0, 0 },
                          { 0, cos(b), - sin(b) },
                          { 0, sin(b), cos(b) } };

  const dom R_z[3][3] = { { cos(c), - sin(c), 0 },
                          { sin(c), cos(c), 0 },
                          { 0, 0, 1 } };

  // actual rotation matrix
  dom R[3][3] = { { 1, 0, 0 },
                  { 0, 1, 0 },
                  { 0, 0, 1 } };

  multiply_matrices( R, R_x );
  multiply_matrices( R, R_y );
  multiply_matrices( R, R_z );
  
  dom3 ret = vec;
  multiply_vector_matrix( ret, R );
  
  return ret;
}

