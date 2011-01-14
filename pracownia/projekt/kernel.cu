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
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>


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

struct arbitrary_functor
{
  typedef thrust::tuple< thrust::counting_iterator< int >, // ix
			 thrust::device_vector< float >::iterator , // x
			 thrust::device_vector< float >::iterator > // y
  ArgTuple;

  thrust::device_ptr< float > randomData;
  thrust::device_ptr< float4 > vertexData;

  __host__ __device__
  void operator()(ArgTuple t)
  {
    float r = sqrtf(randomData[ 2*(*thrust::get<0>(t)) ]);
    float theta = randomData[2*(*thrust::get<0>(t)) + 1] * 2 * ((float)M_PI);

    float x = r * cos(theta);
    float y = r * sin(theta);

    *thrust::get<1>(t) = x;
    *thrust::get<2>(t) = y;

    vertexData[ *thrust::get<0>(t) ] = make_float4(x, y, 0.0f, 1.0f);
  }
};

//__global__ void kernel_random_points(float * randomData, float * outputPoints_x, float * outputPoints_y, float4* pos)
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

//struct DotProduct : public thrust::unary_function<float4*, float, float, float, float>

struct KernelMakePointsDisplay
{

  typedef thrust::tuple< 
                        thrust::device_vector< float >::iterator, // r
                        thrust::device_vector< float >::iterator, // theta
                        thrust::device_vector< float >::iterator, // x
                        thrust::device_vector< float >::iterator, // y
                        float4* // vertex buffer
                       > 
  
          ArgTuple;

  __host__ __device__ void operator()( ArgTuple t )
  {
    float x, y, r, theta;

    r = sqrtf(thrust::get<0>(t));
    theta = 2 * ((float)M_PI) * thrust::get<1>(t);

    x = r * cos(theta);
    y = r * sin(theta);
    
    *thrust::get<2>(t) = x;
    *thrust::get<3>(t) = y;
    *thrust::get<4>(t) = make_float4( x, y, 0.0f, 1.0f );    
  }
};         


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



}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* pos, unsigned int points)
{
  //  std::cout << "cores: " << getCoreCount() << std::endl;

  points *= 1024;

  // generate random numbers
  float * randomPoints;
  static curandGenerator_t gen;
  static bool gen_set = false;
  CUDA_CALL(cudaMalloc((void **)&randomPoints, points * 2 * sizeof(float)));

  if (!gen_set)
    {
      /* Create pseudo-random number generator */
      CURAND_CALL(curandCreateGenerator(&gen,
					CURAND_RNG_PSEUDO_DEFAULT));
  
      /* Set seed */
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(0)));
      gen_set = true;
    }
  CURAND_CALL(curandGenerateUniform(gen, randomPoints, points * 2));
  CUT_CHECK_ERROR("Kernel error");

  // calculate actual points, display them on the screen
  thrust::counting_iterator< float4* > vertexBufferPtr(pos);

  thrust::device_vector<float> outputPoints_x(points);
  thrust::device_vector<float> outputPoints_y(points);

  thrust::device_vector<float> randomPoints_r(randomPoints, randomPoints + points);
  thrust::device_vector<float> randomPoints_theta(randomPoints + points, randomPoints + 2*points);

  thrust::transform( make_tuple( vertexBufferPtr, 
				 outputPoints_x, outputPoints_y,
				 randomPoints_r, randomPoints_theta ),
		     KernelMakePointsDisplay() );
				 
  
  //  dim3 block(512, 1, 1);
  //  dim3 grid(points / 512, 1, 1);
  //  kernel_random_points<<< grid, block>>>(randomPoints, outputPoints_x, outputPoints_y, pos);
  //  CUT_CHECK_ERROR("Kernel error");

  // sort points
  
  thrust::device_vector<float> permutation(points);
  thrust::sequence(permutation.begin(), permutation.end());
  
  // sort from least significant key to most significant keys
  update_permutation(outputPoints_y, permutation);
  update_permutation(outputPoints_x, permutation);

  // permute the key arrays by the final permuation
  apply_permutation(outputPoints_y, permutation);
  apply_permutation(outputPoints_x, permutation);


//  
//  
//  thrust::device_ptr<float2> outputPoints_dev(outputPoints);
// 
//  struct compare_float2 comp2;
//  
//  thrust::device_vector<float2> vec(outputPoints_dev, outputPoints_dev+points);
//  thrust::sort( vec.begin(), vec.end(), comp2 );
  
  //CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(randomPoints));
  CUDA_CALL(cudaFree(outputPoints_x));
  CUDA_CALL(cudaFree(outputPoints_y));
}
