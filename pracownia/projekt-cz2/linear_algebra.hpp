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

__host__ __device__
float3 Normalize( const float3 & Vec )
{
  float len = sqrt(Vec.x * Vec.x + Vec.y * Vec.y + Vec.z * Vec.z);
  return make_float3( Vec.x / len, Vec.y / len, Vec.z / len );
}

template < typename dom >
__host__ __device__
inline
void multiply_matrices(const dom A[3][3], const dom B[3][3], dom C[3][3])
{
  C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] + B[2][0];
  C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] + B[2][1];
  C[0][2] = A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] + B[2][2];

  C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] + B[2][0];
  C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] + B[2][1];
  C[1][2] = A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] + B[2][2];

  C[2][0] = A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] + B[2][0];
  C[2][1] = A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] + B[2][1];
  C[2][2] = A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] + B[2][2];
}

template < typename dom3, typename dom >
__host__ __device__
inline
dom3 multiply_vector_matrix(const dom3 & vec, const dom M[3][3])
{
  dom3 res;
  res.x = vec.x * M[0][0] + vec.y * M[0][1] + vec.z * M[0][2];
  res.y = vec.x * M[1][0] + vec.y * M[1][1] + vec.z * M[1][2];
  res.z = vec.x * M[2][0] + vec.y * M[2][1] + vec.z * M[2][2];
  return res;
}

template < typename dom3 >
__host__ __device__
inline
dom3 translate_point(const dom3 p, const dom3 vec)
{
  dom3 pp;
  pp.x = p.x + vec.x;
  pp.y = p.y + vec.y;
  pp.z = p.z + vec.z;
  return pp;
}


template < typename dom3, typename dom >
__host__ __device__
inline
dom3 rotate_vector(dom3 vec,dom3 euler_angles)
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
  dom R_0[3][3] = { { 1, 0, 0 },
                    { 0, 1, 0 },
                    { 0, 0, 1 } };

  dom R_1[3][3] = { { 1, 0, 0 },
                    { 0, 1, 0 },
                    { 0, 0, 1 } };

  multiply_matrices( R_0, R_x, R_1 ); // yeah, copying R_x would do fine.
  multiply_matrices( R_1, R_y, R_0 );
  multiply_matrices( R_0, R_z, R_1 );
  
  dom3 ret = multiply_vector_matrix( vec, R_1);
  
  return ret;
}

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

template < typename dom3, typename dom >
__host__
inline
dom3 rotate_vector_(dom3 vec,dom3 euler_angles)
{
  namespace ublas = boost::numeric::ublas;

  dom a, b, c; 
  a = euler_angles.x;
  b = euler_angles.y;
  c = euler_angles.z;

  typedef ublas::bounded_matrix< dom, 3, 3 > Mat;
  typedef ublas::bounded_vector< dom, 3 > Vec;

  Mat R_x(3,3);
  R_x(0,0) = 1;
// = { { cos(a), 0, sin(a) },
//          { 0, 1, 0 },
//          { - sin(a), 0, cos(a) } };
 
  const Mat R_y(3,3);// = { { 1, 0, 0 },
                    //{ 0, cos(b), - sin(b) },
                    //{ 0, sin(b), cos(b) } };
  
//  const Mat R_z = { { cos(c), - sin(c), 0 },
//                    { sin(c), cos(c), 0 },
//                    { 0, 0, 1 } };

  const Mat R_z(3,3);

  dom3 ret_;
  Vec ret(3);
  ret[0] = vec.x;
  ret[1] = vec.y;
  ret[2] = vec.z;

  //ret *= (R_z * R_y * R_x);

  std::cout << R_z;

  ret_.x = ret(0);
  ret_.y = ret(1);
  ret_.z = ret(2);

  return ret_;
}

