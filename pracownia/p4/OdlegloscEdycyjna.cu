/* Przykładowe wyjście:

./a.out  slowa.txt fooo bar baz

MAX_L=16
MAX_ARG=2048
TILE=256
Maksymalna długość: 14
Wczytano 795875 słów, 795876 nowych linii
CZAS CAŁKOWITY (CPU): 1152.609985
   1.                 fooo ->                 foot :    1
   2.                  bar ->                  bar :    0
   3.                  baz ->                  baz :    0
CZAS CAŁKOWITY (GPU): 237.882004
   1.                 fooo ->                 foto :    1
   2.                  bar ->                  bar :    0
   3.                  baz ->                  baz :    0

*/

// nvcc -I$CUDA_SDK/C/common/inc -L$CUDA_SDK/C/lib -lcutil_i386 OdlegloscEdycyjna.cu
// Kompilacja:  ^.^
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <locale.h>
#include <iconv.h>

#include <cutil_inline.h>
#include <cuda_runtime.h>

iconv_t iconv_from_utf8;
iconv_t iconv_to_utf8;
texture<char, 1, cudaReadModeElementType> slownik_tex;


template< typename T, typename T2 >
T align_up( T x, T2 y )
{
  return ((x / y) + (x % y ? 1 : 0)) * y;
}

const int MAX_L = 16;  // Maksymalna dlugosc slowa (łącznie z 0 na końcu napisu)
const int MAX_ARG = 16; // Maksymalna liczba argumentów
const int TILE = 256;

__device__ __constant__ char arguments_gpu_const[MAX_ARG * MAX_L];

__device__ __host__ inline int minimum(const int a,const int b,const int c){
  return a<b? (c<a?c:a): (c<b?c:b);
}

char * iconv_wrapper( iconv_t conversion, char * slowo )
{
  static char buf[128][1024];
  static int counter = 0;

  char * slowo_conv = buf[counter];

  size_t iso_len = strlen(slowo);
  size_t conv_len = 1024;

  char * conv = slowo_conv;
  char * iso = slowo;

  iconv( conversion,
         &iso, &iso_len,
         &conv, &conv_len);

  counter++;
  return slowo_conv;
}

char * to_UTF_8( char * slowo )
{
  return iconv_wrapper( iconv_to_utf8, slowo );
}

char * to_ISO_8859_2( char * slowo )
{
  return iconv_wrapper( iconv_from_utf8, slowo );
}

void printOverallResults( int argc, char * proc, char * slownik, double totalTime, dim3 results[MAX_ARG], char * arguments[MAX_ARG] )
{
  printf("CZAS CAŁKOWITY (%s): %2.6f\n", proc, totalTime);
  for(int argNum = 2; argNum < argc; argNum++)
    {
      printf("%4d. %20s -> %20s : %4d\n",
             argNum-1,
             to_UTF_8(arguments[argNum]),
             to_UTF_8(slownik+MAX_L*results[argNum].x),
             results[argNum].y);
    }
}

inline int maximum( int a, int b ){ return a > b ? a : b; }

/////////////// CPU ///////////////
// Odległosc edycyjna Levenshtein'a
__host__ __device__ inline
int OE_CPU(const char *a,const int aN,const char *b,const int bN){
  int gt1C[MAX_L];
  int gt2C[MAX_L];

  int *d1 = gt1C;
  int *d2 = gt2C;

  for (int j=0;j<=bN;j++) d1[j]=j;

  for (int i=1;i<=aN;i++) {
    d2[0] = i;
    for (int j=1;j<=bN;j++) {
      d2[j] = minimum(d1[j  ] + 1,                        // deletion
                      d2[j-1] + 1,                        // insertion
                      d1[j-1] + ((a[i-1]==b[j-1])? 0:1)); // substitution
    }
    d1 = (d1==gt1C)? gt2C:gt1C; // table exchange 1<>2
    d2 = (d2==gt2C)? gt1C:gt2C; // table exchange 1<>2
  }

  return d1[bN];
}

void runCPU( char * slowo, char * slownik, int rozmiar_slownika,
             int * najblizsze_slowo, int * odleglosc)
{
  char *s = slowo;
  int minlCPU=999999, miniCPU;
  for (int i=0; i<rozmiar_slownika; i++)
    {
      char *d=slownik+MAX_L*i;
      int   l = OE_CPU(s,  strlen(s),    d,  strlen(d));
      if (l<minlCPU){   minlCPU=l;  miniCPU=i; }
#ifdef DEBUG
      printf("%02d: %d%s", i, l, (i+1) % 4 ? "\t" : "\n" );
#endif
    }

#ifdef DEBUG
  printf("\n");
#endif

  *najblizsze_slowo = miniCPU;
  *odleglosc = minlCPU;

}

/////////////// GPU ///////////////
__global__
void kernelGPU_OE(int numer_argumentu, char * slownik, int rozmiar_slownika, int dlugosc_slowa, int * reverse_wyniki)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= rozmiar_slownika )
    {
      return;
    }

  char a[MAX_L];
  int aN = 0; // calculated below

  for(int i=0; i < MAX_L; i++)
    {
      char c;
      c = tex1Dfetch( slownik_tex, MAX_L*idx+i);
      //      c = slownik[MAX_L*idx+i];
      if (!c)
        {
          aN = i;
          break;
        }
      a[i] = c;
    }

  int val = OE_CPU( a, aN, (const char*)(arguments_gpu_const+numer_argumentu*MAX_L), dlugosc_slowa );

  reverse_wyniki[val] = idx;
  //atomicMin(&reverse_wyniki[val], idx);
  //atomicCAS( &reverse_wyniki[val], rozmiar_slownika+5, idx);

}

__host__
void runGPU( int numer_argumentu,
             char * slowo, char * slownik_gpu, int rozmiar_slownika,
             int * najblizsze_slowo, int * odleglosc)
{
  int liczba_watkow = align_up(rozmiar_slownika,TILE); // (1+(rozmiar_slownika / TILE)) * TILE;

  const int SPECIAL = rozmiar_slownika+5;

  static int * reverse_wyniki_gpu = NULL;
  if (!reverse_wyniki_gpu)
    cutilSafeCall(cudaMalloc(&reverse_wyniki_gpu, sizeof(int) * (MAX_L+1)));

  int reverse_wyniki[MAX_L+1];
  for(int i = 0; i < MAX_L+1; i++)
    {
      reverse_wyniki[i] = SPECIAL;
    }
  cutilSafeCall(cudaMemcpy(reverse_wyniki_gpu, reverse_wyniki, MAX_L+1, cudaMemcpyHostToDevice));

  // Wywołanie jądra
  dim3 dimGrid(liczba_watkow / TILE);
  dim3 dimBlock(TILE);
  kernelGPU_OE<<<dimGrid, dimBlock>>>(numer_argumentu, slownik_gpu, rozmiar_slownika, strlen(slowo), reverse_wyniki_gpu);
  cutilSafeCall(cudaMemcpy(reverse_wyniki, reverse_wyniki_gpu, MAX_L+1, cudaMemcpyDeviceToHost));

#pragma unroll 16
  for(int i = 0; i < MAX_L+1; i++)
    {
      if (reverse_wyniki[i] != SPECIAL)
        {
          *najblizsze_slowo = reverse_wyniki[i];
          *odleglosc = i;
          return;
        }
    }

  //cutilSafeCall(cudaFree(reverse_wyniki_gpu));
}


int main(int argc, char** argv){
  printf("MAX_L=%d\n", MAX_L);
  printf("MAX_ARG=%d\n", MAX_ARG);
  printf("TILE=%d\n", TILE);

  if ( argc < 3 )
    {
      printf("Za mało argumentów\n");
      return 1;
    }

  if ( argc >= MAX_ARG )
    {
      printf("Za dużo argumentów\n");
      return 1;
    }

  // --------------------------------------------------------------------
  // Zaladuj slownik
  FILE *plik_slownika = fopen(argv[1], "r");  // pierwszy argument !!!
  if (!plik_slownika)
    {
      printf("Nie udało się otworzyć pliku słownika: %s\n", argv[1]);
      return 1;
    }

  // wyliczamy liczbę słów i maksymalną długość
  int max_dl = 0;
  int ilosc = 0;
  while(!feof(plik_slownika))
    {
      char buf[1024];
      int res = fscanf(plik_slownika,"%s",buf);
      int dl = strlen(buf);
#ifdef DEBUG
      printf("Długość słowa: %d\nSłowo: %s\n", dl, buf);
#endif
      max_dl = maximum( max_dl, dl );
      ilosc++;
    }
  printf("Maksymalna długość: %d\n", max_dl);
  assert( max_dl < MAX_L ); // zakładamy że słowa są krótsze niż MAX_L bajty

  /* wczytujemy słownik faktycznie */
  rewind(plik_slownika);
  size_t slownik_size = align_up(sizeof(char) * MAX_L * ilosc, 32);
  char * slownik;
  slownik = (char *) malloc(slownik_size);
  int slownik_rozmiar;
  memset(slownik, 0, slownik_size);

  int cnt=0;
  while(!feof(plik_slownika))
    {
      char buf[1024];
      memset(buf, 0, 1024);
      int res = fscanf(plik_slownika, "%s", buf);
      if ( !strcmp(buf,"") )
        continue;

      memcpy(slownik + cnt * MAX_L, buf, MAX_L);
#ifdef DEBUG
      printf("SŁOWO: %s\n", slownik + cnt * MAX_L);
#endif
      cnt++;
    }
  slownik_rozmiar = cnt;

  printf("Wczytano %d słów, %d nowych linii\n", cnt, ilosc);

  /* konwertujemy argumenty */
  iconv_from_utf8 = iconv_open("ISO-8859-2","UTF-8");
  iconv_to_utf8 = iconv_open("UTF-8","ISO-8859-2");

  char * arguments[MAX_ARG];

  for(int i = 0; i < argc; i++)
    {
      size_t bufLen = sizeof(char) * 1024;
      char * buf = (char *)malloc(bufLen);
      size_t argvLen = strlen(argv[i]);

      char * buf_i = buf;
      char * arg_i = argv[i];

      iconv( iconv_from_utf8,
             &arg_i, &argvLen,
             &buf_i, &bufLen);

      arguments[i] = buf;

#ifdef DEBUG
      printf("argv[%d] = %s(%d) = %s(%d)\n",
             i,
             argv[i], strlen(argv[i]),
             buf, strlen(buf));
#endif
    }

  /* kopiujemy argumenty na GPU */
  //cutilSafeCall(cudaGetSymbolAddress(&arguments_gpu_const, "arguments_gpu"));
  for(int i = 2; i < argc; i++)
    {
      cutilSafeCall(cudaMemcpyToSymbol(arguments_gpu_const, arguments[i], MAX_L,MAX_L*i));
    }

  // --------------------------------------------------------------------
  // -GPU:---------------------------------------------------------------
  // --------------------------------------------------------------------
  /* przesyłanie słownika na GPU */

  char * slownik_GPU;
  cutilSafeCall(cudaMalloc(&slownik_GPU, slownik_size));
  cutilSafeCall(cudaMemcpy(slownik_GPU, slownik, slownik_size, cudaMemcpyHostToDevice));

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();

  // set texture parameters
  slownik_tex.addressMode[0] = cudaAddressModeWrap;
  slownik_tex.filterMode = cudaFilterModePoint;
  slownik_tex.normalized = false;    // access with normalized texture coordinates

  cutilSafeCall(cudaBindTexture(0, slownik_tex, slownik_GPU, channelDesc));


  //texture<char> slownik_Tex;
  //const struct textureReference texRef = slownik_Tex;
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<char>();
  //cutilSafeCall(cudaBindTexture(0, &texRef, (void *)slownik_GPU, &slownik_Tex.channelDesc, slownik_size));

  // Pętla dla CPU
  {
    unsigned int timer = 0;
    double totalTime;
    dim3 results[MAX_ARG];

    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));
    for(int argNum = 2; argNum < argc; argNum++)
      {
        char * slowo = arguments[argNum];
        int numer_slowa = 0, odleglosc = 999;
        runCPU( slowo, slownik, slownik_rozmiar, &numer_slowa, &odleglosc );
        results[argNum].x = numer_slowa;
        results[argNum].y = odleglosc;
      }
    totalTime = cutGetTimerValue(timer);
    printOverallResults( argc, "CPU", slownik, totalTime, results, arguments );
    cutilCheckError(cutDeleteTimer(timer));
  }

  // Pętla dla GPU
  {
    unsigned int timer = 0;
    double totalTime;
    dim3 results[MAX_ARG];

    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));
    for(int argNum = 2; argNum < argc; argNum++)
      {
        char * slowo = arguments[argNum];
        int numer_slowa = 0, odleglosc = 999;
        runGPU( argNum, slowo, slownik_GPU, slownik_rozmiar, &numer_slowa, &odleglosc );
        results[argNum].x = numer_slowa;
        results[argNum].y = odleglosc;
      }
    // zawsze na początku i końcu uruchomienia timera robimy synchronizację. patrz 'best practices guide'
    cutilSafeCall(cudaThreadSynchronize());
    totalTime = cutGetTimerValue(timer);
    printOverallResults( argc, "GPU", slownik, totalTime, results, arguments );
    cutilCheckError(cutDeleteTimer(timer));
  }

  cutilSafeCall(cudaFree(slownik_GPU));
  iconv_close(iconv_from_utf8);
  iconv_close(iconv_to_utf8);

  return 0;


}
