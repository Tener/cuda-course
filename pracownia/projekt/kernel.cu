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

    unsigned int seed = hash(ix+gseed);

    // seed a random number generator
    thrust::default_random_engine rng(seed);

    // create a mapping from random numbers to [0,1)
    thrust::uniform_real_distribution<float> u01(0,1);

    r = sqrtf(u01(rng));
    theta = 2 * ((float)M_PI) * u01(rng);

    x = r * cos(theta);
    y = r * sin(theta);
    
    vbo[ix] = make_float4( x, y, 0.0f, 1.0f );

    return thrust::make_tuple( r, theta );
  }
};         



typedef thrust::tuple< int, float, float > ArgTuple_2; // x, y
typedef thrust::tuple< float, float > ResultTuple_2; // x, y
struct MakePoints : public thrust::unary_function< ArgTuple_2 , ResultTuple_2>
{
  float4 * vbo;
  MakePoints(float4 * vbo) : vbo(vbo) { }

  __host__ __device__ ResultTuple_2 operator()( ArgTuple_2 t )
  {
    int ix = thrust::get<0>(t);

    float r = thrust::get<1>(t);
    float theta = thrust::get<2>(t);

    float x = r * cos(theta);
    float y = r * sin(theta);
    
    vbo[ix] = make_float4( x, y, 0.0f, 1.0f );

    return thrust::make_tuple( x, y );
  }

};


typedef thrust::tuple< float, float > Float2;
typedef thrust::tuple< thrust::device_vector<float>::iterator,
		       thrust::device_vector<float>::iterator > DevVecFloatIterTuple2;
typedef thrust::zip_iterator< DevVecFloatIterTuple2 > DevVecFloatZipIterTuple;


struct CompareX : public thrust::binary_function<Float2,Float2,bool>
{
  float angle;

  CompareX(float angle) : angle(angle) { }

  __host__ __device__ 
  bool operator()(Float2 e1, Float2 e2) const
  {
    return thrust::get<0>(e1) * sin(thrust::get<1>(e1)+angle) < 
           thrust::get<0>(e2) * sin(thrust::get<1>(e2)+angle);
  }
};

struct CompareY : public thrust::binary_function<Float2,Float2,bool>
{
  float angle;

  CompareY(float angle) : angle(angle) { }

  __host__ __device__ 
  bool operator()(Float2 e1, Float2 e2) const
  {
    return thrust::get<0>(e1) * cos(thrust::get<1>(e1)+angle) < 
           thrust::get<0>(e2) * cos(thrust::get<1>(e2)+angle);
  }
};

struct RemoveQuadrilateral
{
  float p0_x, p1_x, p2_x, p3_x, p0_y, p1_y, p2_y, p3_y;

  RemoveQuadrilateral(float p0_x,float p0_y, //p0 - min x
		      float p1_x,float p1_y, //p1 - max x
		      float p2_x,float p2_y, //p2 - min y
		      float p3_x,float p3_y) //p3 - max y
    : p0_x(p0_x), p1_x(p1_x), p2_x(p2_x), p3_x(p3_x), 
      p0_y(p0_y), p1_y(p1_y), p2_y(p2_y), p3_y(p3_y) { }

  __host__ __device__ bool operator()( thrust::tuple< float, float > el )
  {
    float x = thrust::get<0>(el) * cos(thrust::get<1>(el));
    float y = thrust::get<0>(el) * sin(thrust::get<1>(el));

    return x < 0;
  }

};


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* vbo1, int* vbo1_vert_cnt,
					    float4* vbo2, int* vbo2_vert_cnt,
					    unsigned int points)
{
  //  std::cout << "cores: " << getCoreCount() << std::endl;

  points *= 1024;
  *vbo1_vert_cnt = points;

  // calculate points, display them on the screen
  thrust::device_vector<float> oP_r(points);
  thrust::device_vector<float> oP_theta(points);
  thrust::transform( thrust::counting_iterator< int >(0),
		     thrust::counting_iterator< int >(points),
		     thrust::make_zip_iterator(make_tuple( oP_r.begin(), oP_theta.begin())),
		     //		     thrust::make_zip_iterator(make_tuple( oP_x.begin(), oP_y.begin())),
		     KernelMakePointsDisplay(rand(),vbo1) );

  {
    // find extreme points and remove the vast amount of points that lie within the bounding quadrilateral
    float angles[] = { 0, M_PI/2, M_PI/4, -M_PI/4 };    

    for(int i = 0; i < 1; i++)
      {
	DevVecFloatZipIterTuple oP_begin = thrust::make_zip_iterator(make_tuple( oP_r.begin(), oP_theta.begin()));
	DevVecFloatZipIterTuple oP_end = thrust::make_zip_iterator(make_tuple( oP_r.end(), oP_theta.end()));

	float alpha = angles[i];
	thrust::pair< DevVecFloatZipIterTuple, DevVecFloatZipIterTuple > minmax_x, minmax_y;

	minmax_x = thrust::minmax_element( oP_begin,
					   oP_end,
					   CompareX(alpha) );

	minmax_y = thrust::minmax_element( oP_begin,
					   oP_end,
					   CompareY(alpha) );
					   
	DevVecFloatZipIterTuple new_oP_end = thrust::remove_if( oP_begin,
								oP_end, 
								RemoveQuadrilateral( 0, 0, 0, 0, 1, 1, 1, 1 ));


	{
 	  using namespace thrust;
	  std::cerr << "DIFF:" << oP_end - new_oP_end << std::endl;
	  
	  std::cerr << 
	    "MINmax_x: " << get<0>(*minmax_x.first) << " " << get<1>(*minmax_x.first) << std::endl << 
	    "minMAX_x: " << get<0>(*minmax_x.second) << " " << get<1>(*minmax_x.second) << std::endl;
 
	  std::cerr << 
	    "MINmax_y: " << get<0>(*minmax_y.first) << " " << get<1>(*minmax_y.first) << std::endl << 
	    "minmax_y: " << get<0>(*minmax_y.second) << " " << get<1>(*minmax_y.second) << std::endl;
	}

	oP_end = new_oP_end;

	// update number of points
	std::cerr << "P:" << points << std::endl;
	points = oP_end - oP_begin;
	std::cerr << "P:" << points << std::endl;

	oP_r.resize(points);
	oP_theta.resize(points);	
      }

    *vbo2_vert_cnt = points;
  }

  // materialize points (x,y) now
  thrust::device_vector<float> oP_x(points);
  thrust::device_vector<float> oP_y(points);

  thrust::transform( thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(0),
							   oP_r.begin(), 
							   oP_theta.begin())), 
		     thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(points),
							   oP_r.end(), 
							   oP_theta.end())), 
 		     thrust::make_zip_iterator(make_tuple( oP_x.begin(), oP_y.begin())),
 		     MakePoints(vbo2) );  
#if 0
  // sort points

  thrust::device_vector<float> permutation(points);
  thrust::sequence(permutation.begin(), permutation.end());
  
  // sort from least significant key to most significant keys
  update_permutation(oP_y, permutation);
  update_permutation(oP_x, permutation);

  // permute the key arrays by the final permuation
  apply_permutation(oP_y, permutation);
  apply_permutation(oP_x, permutation);
    
  //CURAND_CALL(curandDestroyGenerator(gen));
  //CUDA_CALL(cudaFree(randomPoints));
#endif
}
