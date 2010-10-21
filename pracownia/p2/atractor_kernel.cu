#ifndef _ATRACTOR_KERNEL_H_
#define _ATRACTOR_KERNEL_H_

#include <stdio.h>

#define EPSILON 0.001
#define MAX_ITER 1000

#define CALC_DISTANCE( x1, y1, x2, y2 ) (sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))


__host__ __device__ char calculateAtractor( float a, float b )
{
  int i;
  char result = 1;
  for(i = 0; i < MAX_ITER; i++)
    {
      /* konczymy? */
      if ( CALC_DISTANCE( a, b, 1, 0 ) < EPSILON )  { result = 2; break; }
      if ( CALC_DISTANCE( a, b, 0, 1 ) < EPSILON )  { result = 3; break; }
      if ( CALC_DISTANCE( a, b, -1, 0 ) < EPSILON ) { result = 4; break; }
      if ( CALC_DISTANCE( a, b, 0, -1 ) < EPSILON ) { result = 5; break; }
      /* obliczamy kolejny wyraz ciągu */
      float a_, b_;
      float norm = CALC_DISTANCE( a, b, 0, 0 );
      float norm3 = norm * norm * norm;
      /* wolfram alpha twoim przyjacielem */
      a_ = (3*a + ((a * (a * a - 3 * b * b))/norm3))/4;
      b_ = (3*b + (((b * b - 3 * a * a) * b)/norm3))/4;

      /* full expansion:
      float a2 = a*a;
      float a3 = a*a*a;
      float b2 = b*b;
      float b3 = b*b*b;

+a^3/(4 (a^2+b^2)^3)+(3 a)/4
-(3 a b^2)/(4 (a^2+b^2)^3)
+i (-(3 a^2 b)/(4 (a^2+b^2)^3)+b^3/(4 (a^2+b^2)^3)+(3 b)/4)
      */



      /* nowy zastepuje stary */
      a = a_;
      b = b_;
    }
  /* zwracamy result */
  return result;
}
#undef CALC_DISTANCE

__global__ void kernel_gpu(char * tab, int M, int N, float s)
{
  /* wylicz a i b */
  float a, b;
  float d_a = (2 * s)/(M);
  float d_b = (2 * s)/(N);

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  a = -s + d_a * x;
  b = -s + d_b * y;

  /* wylicz ix */
  int ix = x * N + y; // offset w tablicy

  tab[ix] = calculateAtractor( a, b );
}

__host__ void kernel_cpu(char* tab, int M, int N, float s)
{

  for(int x=0; x < M; x++)
    for(int y=0; y < N; y++)
      {
        /* wylicz a i b */
        float a, b;
        float d_a = (2 * s)/(M);
        float d_b = (2 * s)/(N);

        a = -s + d_a * x;
        b = -s + d_b * y;

        /* wylicz ix */
        int ix = x * N + y; // offset w tablicy
        tab[ix] = calculateAtractor( a, b );
      }
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void GPUAtractor(char* C, int M, int N, float s)
{
    // execute the kernel
//  int mesh_width = 10;
//  int mesh_height = 10;
//  dim3 block(8, 8, 1);
//  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  char * dev_C = 0;
  size_t C_len = sizeof(char) * N * M + 1;

  cudaMalloc ( &dev_C, C_len );

  cudaMemcpy ( dev_C, C, C_len,
               cudaMemcpyHostToDevice );

#define TX 8
#define TY 8

  dim3 threadsPerBlock(TX,TY);
  dim3 numBlocks(M/threadsPerBlock.x, N/threadsPerBlock.y);

#undef TX
#undef TY

  kernel_gpu<<< numBlocks, threadsPerBlock >>>(dev_C, M, N, s);
  cudaThreadSynchronize();

  cudaMemcpy ( C, dev_C, C_len,
               cudaMemcpyDeviceToHost );

  cudaFree( dev_C );
}

extern "C" void CPUAtractor(char* C, int M, int N, float s)
{
  kernel_cpu( C, M, N, s );
}



#endif // #ifndef __ATRACTOR_KERNEL_H__
