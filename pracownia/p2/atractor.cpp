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

// deklaracja funkcji obliczających atraktory
extern "C" void GPUAtractor(char* C, int M, int N, float s);
extern "C" void CPUAtractor(char* C, int M, int N, float s);


// dobre zmienne globalne nie są złe...
int M;
int N;

float s = 2;


// wydrukuj tablicę
char char2repr( char c )
{
  if ( c == 1 ) return '1';
  if ( c == 2 ) return '2';
  if ( c == 3 ) return '3';
  if ( c == 4 ) return '4';
  if ( c == 5 ) return '5';

  return '*';
}

void char2img( FILE * f, char c )
{
  int r, g, b;
  r = 255;
  g = 255;
  b = 255;

  if ( c == 1 ) { r =   0; g =   0; b =   0; }
  if ( c == 2 ) { r =   0; g = 153; b =   0; }
  if ( c == 3 ) { r =   0; g =  76; b = 153; }
  if ( c == 4 ) { r = 153; g =   0; b = 153; }
  if ( c == 5 ) { r = 153; g =  76; b =   0; }

  fprintf(f, "%d %d %d\n", (int)r, (int)g, (int)b );
}


void printResultPPM( char * C, int N, int M )
{
  static int filenum = 0;
  char filename_ppm[1024];
  char filename_png[1024];
  sprintf(filename_ppm,"file_out_%d.ppm", filenum);
  sprintf(filename_png,"file_out_%d.png", filenum);
  filenum++;
  FILE * file = fopen(filename_ppm,"w");

  fprintf(file, "P3\n");
  fprintf(file, "%d %d\n", M, N );
  fprintf(file, "255\n");

  for(int i = 0; i < N; i++)
    {
    for(int j = 0; j < M; j++)
      {
        char2img(file, C[i*M + j]);
      }
    }

  fclose(file);

  char command[1024];
  sprintf(command,"convert %s %s", filename_ppm, filename_png);
  system(command);
  unlink(filename_ppm);
}

void printResultChar( char * C, int N, int M )
{
  for(int i = 0; i < N; i++)
    {
      printf("%02d: ", i);
    for(int j = 0; j < M; j++)
      {
        printf("%c", char2repr(C[i*M + j]));
      }
    printf("\n");
    }
}

void printResult( char * C, int N, int M )
{
  if ( getenv( "ATROUT") )
    {
      if (!strcmp( getenv( "ATROUT"), "text" ) )
        {
          printResultChar( C, N, M ); return;
        }

      if (!strcmp( getenv( "ATROUT"), "ppm" ) )
        {
          printResultPPM( C, N, M ); return;
        }
    }

  // default case
  printResultChar( C, N, M );
}

const int CPU = 1;
const int GPU = 2;

double timevaldiff(struct timeval starttime, struct timeval finishtime)
{
  double msec=0;
  msec+=(finishtime.tv_usec-starttime.tv_usec);
  msec+=(finishtime.tv_sec-starttime.tv_sec)*1000000;
  return (msec/1000000);
}

void run( int proc )
{
  bool write_img = true;
  //  write_img = ( (N * M) < (5000 * 5000));

  size_t C_len = sizeof(char) * N * M + 1;
  char * C = (char *)malloc(C_len);
  memset( C, '\0', C_len);
  struct timeval tv_start, tv_stop;

  gettimeofday(&tv_start, NULL);

  if ( proc == CPU )
    CPUAtractor( C, N, M, s );

  if ( proc == GPU )
    GPUAtractor( C, N, M, s );

  gettimeofday(&tv_stop, NULL);

  if ( write_img )
    printResult( C, N, M );
  printf("%s time: %f\n",
         proc == CPU ? "CPU" : "GPU",
         timevaldiff(tv_start,tv_stop));
  free( C );
}



int main(int argn, char ** argv)
{

  if (argn != 3)
    {
      printf("Podaj 2 argumenty: M i N\n");
      return 1;
    }

  sscanf(argv[1], "%d", &M);
  sscanf(argv[2], "%d", &N);

  printf("M=%d N=%d\n", M, N);

  run( CPU );
  run( GPU );

  return 0;
}
