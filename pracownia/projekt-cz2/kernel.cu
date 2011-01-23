#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>

// CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>
#include <curand_kernel.h>

// thrust
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "common.hpp"
#include "graphics.hpp"

__host__ __device__
inline uint ARGB( uint a, uint r, uint g, uint b ){ return (((((a << 4) + r) << 4) + g) << 4) + b;}

struct TracePoint
{
  int w; int h; int ix_w;
  TracePoint(int w, int h, int ix_w) : w(w), h(h), ix_w(ix_w) { };

  __device__ 
  uint operator()( int ix_h )
  {
    //xsvbo[vbo_ix] = make_float4(ix_w, ix_h, 0.0f, 1.0f);
    
    return 0x12345678; //ARGB( ix_h, ix_w, ((float)w) / ((float)h), 128 );
  }
};


extern "C" void launch_raytrace_kernel(uint * pbo, int w, int h)
{
  for(int ix_w = 0; ix_w < w; ix_w++)
    {
      thrust::transform( thrust::make_counting_iterator< uint >(0),
                         thrust::make_counting_iterator< uint >(h),
                         thrust::device_ptr< uint >(pbo + w * ix_w),
                         TracePoint(w,h,ix_w) );
    }

}
