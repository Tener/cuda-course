/*

N=64
CPU time: 0.001416
GPU(T=8) time: 0.101907
GPU(T=16) time: 0.000406
N=256

CPU time: 0.041779
GPU(T=8) time: 0.040971
GPU(T=16) time: 0.026334

N=1024
CPU time: 11.634661
GPU(T=8) time: 1.623204
GPU(T=16) time: 0.836923

 */




// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>

// includes
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <rendercheck_gl.h>
//#include <cutil_gl_error.h>

#include <openssl/md5.h>

// deklaracja funkcji
extern "C" void GPUMatrixMul_T8 (float *A, float *B, float *C, int N);
extern "C" void GPUMatrixMul_T16 (float *A, float *B, float *C, int N);



double timevaldiff(struct timeval starttime, struct timeval finishtime)
{
  double msec=0;
  msec+=(finishtime.tv_usec-starttime.tv_usec);
  msec+=(finishtime.tv_sec-starttime.tv_sec)*1000000;
  return (msec/1000000);
}

void MulMatrixCPU(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float suma = 0;
            for (int k = 0; k < N; ++k)
                suma +=  A[i * N + k]  * B[k * N + j];
            C[i * N + j] = suma;
        }
}


void run( int tile,   float * A,  float * B,  float * C,  int N )
{
  struct timeval tv_start, tv_stop;

  gettimeofday(&tv_start, NULL);

#define CALL( fun ) fun( A, B, C, N );

  switch ( tile ) {
  case    0: CALL(MulMatrixCPU); break;
  case    8: CALL(GPUMatrixMul_T8); break;
  case   16: CALL(GPUMatrixMul_T16); break;
  }

#undef CALL

  gettimeofday(&tv_stop, NULL);

  if ( tile )
    {
      printf("GPU(T=%d) time: %f\n",
             tile,
             timevaldiff(tv_start,tv_stop));
    }
  else // CPU
    {
      printf("CPU time: %f\n",
             timevaldiff(tv_start,tv_stop));
    }

}

void checksum( const void * buf, size_t len )
{
  unsigned char md[ 16 ];
  MD5( (const unsigned char*) buf, len, md );
  printf("MD5: 0x");
  for(int i = 0; i < 16; i++)
    {
      printf("%x", md[i]);
    }
  printf("\n");
}

void experiment(int N)
{
  size_t N_len;
  float * A;
  float * B;
  float * C;

  printf("N=%d\n", N);

  N_len = sizeof(float) * N * N;

  // alokujemy macierze i zapełniamy je

  A = (float *)malloc(N_len);
  B = (float *)malloc(N_len);
  C = (float *)malloc(N_len);

  for(int i = 0; i < N*N; i++)
    {
      A[i] = ((float)drand48()*20.0) - 10.0;
      B[i] = ((float)drand48()*20.0) - 10.0;
      C[i] = 0;
    }

  // uruchamiamy procedury

  run(    0, A, B, C, N ); checksum( C, N_len );
  run(    8, A, B, C, N ); checksum( C, N_len );
  run(   16, A, B, C, N ); checksum( C, N_len );

  // zwalniamy macierze

  free( A );
  free( B );
  free( C );
}


int main()
{
  srand48( time(NULL) );
  experiment( 64 );
  experiment( 256 );
  experiment( 1024 );

  return 0;
}

