#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>
#include <curand_kernel.h>

// utilities and system includes
#include <shrUtils.h>

// thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


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

__global__ void kernel_random_points(float * randomData, 
				     float * outputPoints_x, 
				     float * outputPoints_y, 
				     float4 * pos)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  
  float r = sqrtf(randomData[2*x]);
  float theta = randomData[2*x+1] * 2 * ((float)M_PI);
  
  float x_ = r * cos(theta);
  float y_ = r * sin(theta);

  outputPoints_x[x] = x_;
  outputPoints_y[x] = y_;
 
  // write output vertex
  pos[x] = make_float4(x_, y_, 0.0f, 1.0f);
}


template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // temporary storage for keys
    KeyVector temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}

template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // copy keys to temporary vector
    KeyVector temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}


float3 getCoreCount()
{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0); //device number 0
	return make_float3( deviceProp.multiProcessorCount,
			    ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			    deviceProp.multiProcessorCount * ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
}



typedef thrust::tuple< int > ArgTuple_1; // ix
typedef thrust::tuple< float, float > ResultTuple_1; // x, y

struct KernelMakePointsDisplay : public thrust::unary_function< ArgTuple_1 , ResultTuple_1>
{
  //float * randomPoints;
  float4 * vbo;
  unsigned int gseed;

  __host__ __device__
  unsigned int hash(unsigned int a)
  {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }

  KernelMakePointsDisplay(unsigned int gseed, float4 * vbo)
    : gseed(gseed), vbo(vbo) {}

  __host__ __device__ ResultTuple_1 operator()( ArgTuple_1 t )
  {
    float x, y, r, theta;
    int ix = thrust::get<0>(t);

    unsigned int seed = gseed + hash(ix);

    // seed a random number generator
    thrust::default_random_engine rng(seed);

    // create a mapping from random numbers to [0,1)
    thrust::uniform_real_distribution<float> u01(0,1);

    r = sqrtf(u01(rng));
    theta = 2 * ((float)M_PI) * u01(rng);

    x = r * cos(theta);
    y = r * sin(theta);
    
    vbo[ix] = make_float4( x, y, 0.0f, 1.0f );

    return thrust::make_tuple( x, y );
  }
};         

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* vbo, unsigned int points)
{
  //  std::cout << "cores: " << getCoreCount() << std::endl;

  points *= 1024;

  // calculate actual points, display them on the screen
  thrust::device_vector<float> oP_x(points);
  thrust::device_vector<float> oP_y(points);
  thrust::device_vector<float> oP_r(points);
  thrust::device_vector<float> oP_theta(points);
  thrust::transform( thrust::counting_iterator< int >(0),
		     thrust::counting_iterator< int >(points),
		     thrust::make_zip_iterator(make_tuple( oP_x.begin(), oP_y.begin())),
		     KernelMakePointsDisplay(rand(),vbo) );

  // find extreme points and remove the vast amount of points that lie within the bounding square

  // sort points
  thrust::device_vector<float> permutation(points);
  thrust::sequence(permutation.begin(), permutation.end());
  
  // sort from least significant key to most significant keys
  update_permutation(outputPoints_y, permutation);
  update_permutation(outputPoints_x, permutation);

  // permute the key arrays by the final permuation
  apply_permutation(outputPoints_y, permutation);
  apply_permutation(outputPoints_x, permutation);

  // {
  //   thrust::host_vector< float > h_outputPoints_x( outputPoints_x );
  //   thrust::host_vector< float > h_outputPoints_y( outputPoints_y );
    
  //   for(int i = 0; i < points; i++)
  //     {
  // 	std::cout << "(" << h_outputPoints_x[i] << ", " << h_outputPoints_y[i] << ")" << std::endl;
  //     }

  // }
    
      

  //CURAND_CALL(curandDestroyGenerator(gen));
  //CUDA_CALL(cudaFree(randomPoints));
  //CUDA_CALL(cudaFree(outputPoints_x));
  //CUDA_CALL(cudaFree(outputPoints_y));
}
