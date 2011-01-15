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



typedef thrust::tuple< float, float, float4* > ArgTuple_1; // r, theta, ptr to vertex buffer object
typedef thrust::tuple< float, float > ResultTuple_1; // x, y

struct KernelMakePointsDisplay : public thrust::unary_function< ArgTuple_1 , ResultTuple_1>
{
  __host__ __device__ ResultTuple_1 operator()( ArgTuple_1 t )
  {
    float x, y, r, theta;

    r = sqrtf(thrust::get<0>(t));
    theta = 2 * ((float)M_PI) * (thrust::get<1>(t));

    x = r * cos(theta);
    y = r * sin(theta);
    
    *thrust::get<2>(t) = make_float4( x, y, 0.0f, 1.0f );

    return thrust::make_tuple( x, y );//, ;
  }
};         

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* vbo, unsigned int points)
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
  thrust::device_vector<float> outputPoints_x(points);
  thrust::device_vector<float> outputPoints_y(points);

  thrust::device_ptr< float > randomPoints_devPtr(randomPoints);

  thrust::device_vector<float> randomPoints_r(randomPoints_devPtr, randomPoints_devPtr + points);
  thrust::device_vector<float> randomPoints_theta(randomPoints_devPtr + points, randomPoints_devPtr + 2*points);

  thrust::counting_iterator<float4*> vbo_cnt_iter(vbo);

  thrust::transform( thrust::make_zip_iterator(make_tuple( randomPoints_r.begin(), randomPoints_theta.begin(), vbo_cnt_iter )),
		     thrust::make_zip_iterator(make_tuple( randomPoints_r.end(), randomPoints_theta.end(), vbo_cnt_iter+points)),
		     thrust::make_zip_iterator(make_tuple( outputPoints_x.begin(), outputPoints_y.begin())),
		     KernelMakePointsDisplay() );
				 
  
  //  dim3 block(512, 1, 1);
  //  dim3 grid(points / 512, 1, 1);
  //  kernel_random_points<<< grid, block>>>(randomPoints, outputPoints_x, outputPoints_y, vbo);
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

  //CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(randomPoints));
  //CUDA_CALL(cudaFree(outputPoints_x));
  //CUDA_CALL(cudaFree(outputPoints_y));
}
