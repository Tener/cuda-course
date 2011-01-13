#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cutil.h>
#include <curand_kernel.h>

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

__global__ void kernel_random_points(float * randomData, float4* pos)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  
  float r = sqrtf(randomData[x]);
  float theta = randomData[x+1] * 2 * ((float)M_PI);
  
  float x_ = r * cos(theta);
  float y_ = r * sin(theta);
 
  // write output vertex
  pos[x] = make_float4(x_, y_, 0.0f, 1.0f);

}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* pos, unsigned int points)
{
  // create random numbers
  float * randomPoints;
  static curandGenerator_t gen;
  static bool gen_set = false;
  
  CUDA_CALL(cudaMalloc((void **)&randomPoints, points * 2 * 1024 * sizeof(float)));

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
  CURAND_CALL(curandGenerateUniform(gen, randomPoints, points * 2 * 1024));

  CUT_CHECK_ERROR("Kernel error");
  
  // execute the kernel
  dim3 block(512, 1, 1);
  dim3 grid(points * 2, 1, 1);
  kernel_random_points<<< grid, block>>>(randomPoints, pos);

  CUT_CHECK_ERROR("Kernel error");

  //CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_CALL(cudaFree(randomPoints));
}
