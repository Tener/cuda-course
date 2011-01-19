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
#include <thrust/partition.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cpu.hpp"

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
  float x0, x1, x2, x3, y0, y1, y2, y3;

  // http://stackoverflow.com/questions/243945/calculating-a-2d-vectors-cross-product
  __host__ __device__ inline
  float crossProduct( float x0, float y0, float x1, float y1 )
  {
    return x0*y1 - y0*x1;
  }

  RemoveQuadrilateral(float x0,float y0, //min x
		      float x1,float y1, //max x
		      float x2,float y2, //min y
		      float x3,float y3) //max y
    : x0(x0), x1(x1), x2(x2), x3(x3), 
      y0(y0), y1(y1), y2(y2), y3(y3) { }

  __host__ __device__ 
  bool operator()( thrust::tuple< float, float > el )
  {
    float x = thrust::get<0>(el) * cos(thrust::get<1>(el));
    float y = thrust::get<0>(el) * sin(thrust::get<1>(el));

    // (A − B) × (p − B) > 0
    // http://stackoverflow.com/questions/243945/calculating-a-2d-vectors-cross-product
    // { x0-x1, y0-y1 } × { x-x1, y-y1 };
    bool above_0_1 = (x0-x1)*(y-y1) - (y0-y1)*(x-x1) < 0;
    bool above_1_2 = (x1-x2)*(y-y2) - (y1-y2)*(x-x2) < 0;
    bool above_2_3 = (x2-x3)*(y-y3) - (y2-y3)*(x-x3) < 0;
    bool above_3_0 = (x3-x0)*(y-y0) - (y3-y0)*(x-x0) < 0;

    return !(above_0_1 || above_1_2 || above_2_3 || above_3_0);
}

};


struct Visualize
{
  float4 * vbo;
  Visualize(float4 * vbo) : vbo(vbo) { }
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    vbo[thrust::get<0>(t)] = make_float4( thrust::get<1>(t), thrust::get<2>(t), 0.0f, 1.0f );
  }
};

struct Visualize2
{
  thrust::device_ptr< float4 > vbo;
  Visualize2(float4 * vbo) : vbo(vbo) { }
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    vbo[thrust::get<0>(t)] = make_float4( thrust::get<0>(thrust::get<1>(t)), 
					  thrust::get<1>(thrust::get<1>(t)),
					  0.0f, 1.0f);
  }
};


struct PolarIntoPlanar
{
  template <typename Tuple>
  __host__ __device__
  Tuple operator()(Tuple t)
  {
    float r = thrust::get<0>(t);
    float th = thrust::get<1>(t);
    return thrust::make_tuple( r * cos(th), r * sin(th) );
  }
};

struct EarlyPartition
{
  float x0, y0, x1, y1;
  EarlyPartition(float x0, float y0, float x1, float y1) : x0(x0), y0(y0), x1(x1), y1(y1) 
  {
  }

  template <typename Tuple>
  __host__ __device__
  bool operator()(Tuple t)
  {
    float x = thrust::get<0>(t);
    float y = thrust::get<1>(t);
    return (x0-x1)*(y-y1) - (y0-y1)*(x-x1) >= 0;
  }
};

struct mkPoint : public thrust::unary_function< thrust::tuple< float, float > , Point >
{
  template <typename Tuple>
  __host__ __device__
  Point operator()(Tuple t)
  {
    //int ix = thrust::get<0>(t);
    float x = thrust::get<0>(t);
    float y = thrust::get<1>(t);
    //vec[ix] = Point(x,y);
    return Point(x,y);
  }
};


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* vbo1, int* vbo1_vert_cnt,
					    float4* vbo2, int* vbo2_vert_cnt,
					    unsigned int points)
{
  //  sleep(1);

  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
    
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);


  //  std::cout << "cores: " << getCoreCount() << std::endl;
  //unsigned int points_orig = points;

  points *= 1024;
  *vbo1_vert_cnt = points;

  // calculate points, display them on the screen
  thrust::device_vector<float> oP_r(points);
  thrust::device_vector<float> oP_theta(points);
  thrust::transform( thrust::counting_iterator< int >(0),
		     thrust::counting_iterator< int >(points),
		     thrust::make_zip_iterator(make_tuple( oP_r.begin(), oP_theta.begin())),
		     KernelMakePointsDisplay(rand(),vbo1) );  

  {
    // first points on convex hull - taken from quadrilaterals
    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    thrust::host_vector<float> h_t;
    thrust::host_vector<float> h_r;
    thrust::host_vector< thrust::tuple< float, float > > h_rt;

    // find extreme points and remove the vast amount of points that lie within the bounding quadrilateral
    float angles[] = { 0 * (M_PI/8), 1 * (M_PI/8), 2 * (M_PI/8), 3 * (M_PI/8) };


    for(int i = 0; i < 4; i++)
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
				
	float r0 = thrust::get<0>(*minmax_x.first),  t0 = thrust::get<1>(*minmax_x.first);
	float r1 = thrust::get<0>(*minmax_x.second), t1 = thrust::get<1>(*minmax_x.second);
	float r2 = thrust::get<0>(*minmax_y.first),  t2 = thrust::get<1>(*minmax_y.first);
	float r3 = thrust::get<0>(*minmax_y.second), t3 = thrust::get<1>(*minmax_y.second);

	float x0 = r0 * cos(t0), y0 = r0 * sin(t0);
	float x1 = r1 * cos(t1), y1 = r1 * sin(t1);
	float x2 = r2 * cos(t2), y2 = r2 * sin(t2);
	float x3 = r3 * cos(t3), y3 = r3 * sin(t3);
	
	h_rt.push_back( thrust::make_tuple( r0, t0 ) ); h_t.push_back( t0 );
	h_rt.push_back( thrust::make_tuple( r1, t1 ) ); h_t.push_back( t1 ); 
	h_rt.push_back( thrust::make_tuple( r2, t2 ) ); h_t.push_back( t2 ); 
	h_rt.push_back( thrust::make_tuple( r3, t3 ) ); h_t.push_back( t3 ); 

	if ( 0 )
	{
 	  using namespace thrust;

	  std::cerr << 
	    "MINmax_x: " << get<0>(*minmax_x.first)  << " " << get<1>(*minmax_x.first)  << std::endl << 
	    "minMAX_x: " << get<0>(*minmax_x.second) << " " << get<1>(*minmax_x.second) << std::endl;
 
	  std::cerr << 
	    "MINmax_y: " << get<0>(*minmax_y.first)  << " " << get<1>(*minmax_y.first)  << std::endl << 
	    "minmax_y: " << get<0>(*minmax_y.second) << " " << get<1>(*minmax_y.second) << std::endl;
	}


	DevVecFloatZipIterTuple new_oP_end = thrust::remove_if( oP_begin,
								oP_end, 
								RemoveQuadrilateral( x0, y0, 
										     x2, y2, 
										     x1, y1, 
										     x3, y3 ));

	//std::cerr << "DIFF:" << oP_end - new_oP_end << std::endl;
	oP_end = new_oP_end;

	// update number of points
	//std::cerr << "P:" << points << std::endl;
	points = oP_end - oP_begin;
	//std::cerr << "P:" << points << std::endl;

	oP_r.resize(points);
	oP_theta.resize(points);	
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
    
    //
    {
      h_x.reserve(h_rt.size());
      h_y.reserve(h_rt.size());
      
      thrust::sort_by_key( h_t.begin(), h_t.end(), h_rt.begin() );
      thrust::transform( h_rt.begin(), 
			 h_rt.end(), 
			 thrust::make_zip_iterator(thrust::make_tuple(h_x.begin(), h_y.begin())),
			 PolarIntoPlanar()
			 );
    }

    DevVecFloatZipIterTuple begin = thrust::make_zip_iterator(make_tuple( oP_x.begin(), oP_y.begin()));
    DevVecFloatZipIterTuple begin_0 = thrust::make_zip_iterator(make_tuple( oP_x.begin(), oP_y.begin()));
    DevVecFloatZipIterTuple end = thrust::make_zip_iterator(make_tuple( oP_x.end(), oP_y.end()));
    thrust::host_vector< DevVecFloatZipIterTuple > ends;

    for(int i = 0; i < 16; i++)
      {
	float x0, y0, x1, y1;

	x0 = h_x[i];
	y0 = h_y[i];

	x1 = h_x[(i+1) % 16];
	y1 = h_y[(i+1) % 16];

	//float r0, t0, r1, t1;
// 	r0 = thrust::get<0>(h_rt[i]);
// 	t0 = thrust::get<1>(h_rt[i]);
// 	r1 = thrust::get<0>(h_rt[(i+1) % 16]);
// 	t1 = thrust::get<1>(h_rt[(i+1) % 16]);

 //	//	if ( i <= (points_orig % 16) )
 //	  {
 //	    std::cout << "i: " << i << std::endl;
 //
 //	    std::cout << "x0: " << x0 << " y0: " << y0 << " r0: " << r0 << " t0: " << t0 << std::endl;
 //	    std::cout << "x1: " << x1 << " y1: " << y1 << " r1: " << r1 << " t1: " << t1 << std::endl;
 //	    
 //	    glColor3f( 0.1, 0.3, 0.3 );
 //	    glPointSize(10);
 //	    glLineWidth(3);
 //	    glBegin(GL_LINES);
 //	    glVertex2f( x0, y0 );
 //	    glVertex2f( x1, y1 );
 //	    glEnd();
 //	    
 //	  }
	begin = thrust::partition( begin, end, EarlyPartition(x0, y0, x1, y1) );
	ends.push_back(begin);
      }
    
    points = begin - begin_0;
    oP_x.resize(points);
    oP_y.resize(points);
    *vbo2_vert_cnt = points;

    thrust::for_each( thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(0),
							    oP_x.begin(),
							    oP_y.begin())),
		      thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(points),
							    oP_x.end(),
							    oP_y.end())),
		      Visualize(vbo2) );

    thrust::host_vector< thrust::tuple< float, float > > vpoints;
    std::vector< Point > std_vpoints;
    std_vpoints.resize(points);
    vpoints.resize(points);
    
    thrust::copy( thrust::make_zip_iterator(thrust::make_tuple(oP_x.begin(),
							       oP_y.begin())),
		  thrust::make_zip_iterator(thrust::make_tuple(oP_x.end(),
							       oP_y.end())),
		  vpoints.begin()
		  );

    std::cout << "Number of points after cut: " << points << " " << oP_x.size() << " " << oP_y.size() << " " << vpoints.size() << std::endl;
    
    thrust::host_vector< thrust::tuple< float, float > > chull = hull::alg::cpu::convex_hull(vpoints);
    
    thrust::for_each( thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(0),
							    chull.begin())),
		      thrust::make_zip_iterator(make_tuple( thrust::counting_iterator< int >(chull.size()),
							    chull.end())),
		      Visualize2(vbo2) );
    
    *vbo2_vert_cnt = chull.size();
    std::cout << "Convex hull size: " << chull.size() << std::endl;
  }

  cudaThreadSynchronize();
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cout << "Total time: " << elapsed_time << " milliseconds" << std::endl;


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
