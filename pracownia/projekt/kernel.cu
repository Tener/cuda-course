#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void kernel(float4* pos, unsigned int points)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

  curandState state;
  curand_init(blockIdx.x, x, 0, &state);
  
  float r = sqrt(curand_uniform(&state));
  float theta = curand_uniform(&state) * 2 * M_PI;
  
  float x_ = r * cos(theta);
  float y_ = r * sin(theta);

  // write output vertex
  pos[x] = make_float4(x_, y_, 0.0f, 1.0f);

}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel_random_points(float4* pos, unsigned int points)
{
    // execute the kernel
    dim3 block(256, 1, 1);
    dim3 grid(points * (1024 / 256), 1, 1);
    kernel<<< grid, block>>>(pos, points);
}
