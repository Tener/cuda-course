#ifndef _ATRACTOR_KERNEL_H_
#define _ATRACTOR_KERNEL_H_

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
      // specjalny przypadek dla (0,0) -- odkomentować jakby były problemy z SIGFPE
/*    if ( CALC_DISTANCE( a, b, 0, 0 ) < EPSILON )  { result = 1; break; }    */
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
  a = (s / (2*M))*threadIdx.x - s;
  b = (s / (2*N))*threadIdx.y - s;

  /* wylicz ix */
  int ix = threadIdx.x * M + threadIdx.y; // offset w tablicy

  tab[ix] = calculateAtractor( a, b );
}

//__host__ void kernel_cpu(char* C, int M, int N, float s

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void GPUAtractor(char* C, int M, int N, float s)
{
    // execute the kernel
//  int mesh_width = 10;
//  int mesh_height = 10;
//  dim3 block(8, 8, 1);
//  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  char * arg = 0;
  size_t C_len = sizeof(char) * N * M + 1;

  cudaMalloc ( &arg, C_len );

  cudaMemcpy ( arg, C, C_len,
               cudaMemcpyHostToDevice );

  kernel_gpu<<< N, M >>>(C, M, N, s);

  cudaMemcpy ( C, arg, C_len,
               cudaMemcpyDeviceToHost );
}

extern "C" void CPUAtractor(char* C, int M, int N, float s)
{
    
}



#endif // #ifndef __ATRACTOR_KERNEL_H__
