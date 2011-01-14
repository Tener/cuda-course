#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include <curand_kernel.h>

// utilities and system includes
#include <shrUtils.h>

// CUDA-C includes
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <algorithm>

#include <iostream>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error (%d) at %s:%d\n", __FILE__,__LINE__);	\
      return ;}} while(0)

#define CURAND_CALL(x) CURAND_CALL_( (x), __FILE__, __LINE__) 

inline void CURAND_CALL_(curandStatus st, char * file, int line )
{
  if (st != CURAND_STATUS_SUCCESS)
    {
      printf("Error (%d) at %s:%d\n", st, file, line);
    }
}

__global__ void kernel_random_points(float * randomData, float2 * outputPoints, float4* pos)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  
  float r = sqrtf(randomData[2*x]);
  float theta = randomData[2*x+1] * 2 * ((float)M_PI);
  
  float x_ = r * cos(theta);
  float y_ = r * sin(theta);

  outputPoints[x].x = x_;
  outputPoints[x].y = y_;
 
  // write output vertex
  pos[x] = make_float4(x_, y_, 0.0f, 1.0f);

}


struct compare_float2
{
  __host__ __device__ bool operator() (const float2 & e1, const float2 & e2)
  {
    if (e1.y < e2.y)
      return true;
    if (e1.y > e2.y)
      return false;
    return e1.x < e2.x;
  }
};

struct compare_float2 comp2;

/*
__global__ void convex_hull(vector<float2> P)
{
  int n = P.size(), k = 0;
  vector<float2> H(2*n);
  
  // Sort points lexicographically - this should be done on the 'outside'
  //sort(P.begin(), P.end(), comp2);
    
  // Build lower hull
  for (int i = 0; i < n; i++) {
    while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
    H[k++] = P[i];
  }
    
  // Build upper hull
  for (int i = n-2, t = k+1; i >= 0; i--) {
    while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
    H[k++] = P[i];
  }
    
  H.resize(k);
  return H;
}
*/

int getCoreCount()
{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0); //device number 0
	return ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* pos, unsigned int points)
{
  //  std::cout << "cores: " << getCoreCount() << std::endl;

  points *= 1024;

  // create random numbers
  float * randomPoints;
  float2 * outputPoints;
  static curandGenerator_t gen;
  static bool gen_set = false;
  
  CUDA_CALL(cudaMalloc((void **)&randomPoints, points * 2 * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&outputPoints, points * sizeof(float2)));

  if (!gen_set)
    {
      /* Create pseudo-random number generator */
      CURAND_CALL(curandCreateGenerator(&gen,
					CURAND_RNG_PSEUDO_DEFAULT));
  
      /* Set seed */
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(0)));
      gen_set = true;
    }
  /* Generate n floats on device */
  CURAND_CALL(curandGenerateUniform(gen, randomPoints, points * 2));

  CUT_CHECK_ERROR("Kernel error");
  
  // execute the kernel
  dim3 block(512, 1, 1);
  dim3 grid(points / 512, 1, 1);
  kernel_random_points<<< grid, block>>>(randomPoints, outputPoints, pos);

  CUT_CHECK_ERROR("Kernel error");

  // sort points
  
  thrust::device_ptr<float2> outputPoints_dev(outputPoints);

  struct compare_float2 comp2;
  
  thrust::device_vector<float2> vec(outputPoints_dev, outputPoints_dev+points);
  thrust::sort( vec.begin(), vec.end(), comp2 );
  


  //CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(randomPoints));
  CUDA_CALL(cudaFree(outputPoints));
}
