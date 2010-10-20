// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

#define EPSILON 0.001

// deklaracja funkcji obliczających atraktory
extern "C" void GPUAtractor(char* C, int M, int N, float s);
extern "C" void CPUAtractor(char* C, int M, int N, float s);

// wydrukuj tablicę

char char2repr( char c )
{
  if ( c == 0 ) return '0';
  if ( c == 1 ) return '1';
  if ( c == 2 ) return '2';
  if ( c == 3 ) return '3';
  if ( c == 4 ) return '4';
  if ( c == 5 ) return '5';
  if ( c == 6 ) return '6';

  return '*';
}

void printResult( char * C, int N, int M )
{
  for(int i = 0; i < N; i++)
    {
      printf("%d: ", i);
    for(int j = 0; j < M; j++)
      {
        printf("%c", char2repr(C[i*M + j]));
      }
    printf("\n");
    }
}

int main(int argn, char ** argv)
{
  int M;
  int N;

  float s = 2;

  if (argn != 3)
    {
      printf("Podaj 2 argumenty: M i N\n");
      return 1;
    }

  sscanf(argv[1], "%d", &M);
  sscanf(argv[2], "%d", &N);

  printf("M=%d N=%d\n", M, N);

  size_t C_len = sizeof(char) * N * M + 1;
  char * C = (char *)malloc(C_len);
  memset( C, '\0', C_len);

  CPUAtractor( C, N, M, s ); printResult( C, N, M );
  GPUAtractor( C, N, M, s ); printResult( C, N, M );

  return 0;
}
