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

