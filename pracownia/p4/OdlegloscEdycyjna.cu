/*

./a.out slowa.txt testowość przewlokły woretewr żywotnikowiek gżegżułka

*/

// nvcc -I$CUDA_SDK/C/common/inc -L$CUDA_SDK/C/lib -lcutil_i386 OdlegloscEdycyjna.cu
// Kompilacja:  ^.^
#include <stdlib.h>
#include <stddef.h>
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
// env_porcelain: kontrolowane przez zmienną środowiskową PORCELAIN.
// jeżeli jest ustawiona, to wyjście programu jest w formie łatwej do sparsowania przez programy.
bool env_porcelain;
// SKIP_CPU=1 -> rozmiar słownika dla CPU ustawiany jest na małą liczbę (np. 16), przez co nie CPU nie spowalnia nam obliczeń
bool env_skip_cpu;

template< typename T, typename T2 >
T align_up( T x, T2 y )
{
  return ((x / y) + (x % y ? 1 : 0)) * y;
}

#define SPECIAL( x ) (x + 5) // specjalna wartość do oznaczania niezajętych jeszcze pól
const int MAX_L = 16;  // Maksymalna dlugosc slowa (łącznie z 0 na końcu napisu)
const int MAX_ARG = 32; // Maksymalna liczba argumentów

#ifndef WORDS_PER_THREAD
const int WORDS_PER_THREAD = 1;
#endif

#ifndef TILE
const int TILE = 64;
#endif

typedef struct slownik_entry
{
  char dlugosc;
  char common;
  char slowo[MAX_L];
} slownik_entry;

__device__ __constant__ char arguments_gpu_const[MAX_ARG * MAX_L];

__device__ __host__ inline int minimum3(const int a,const int b,const int c){
  return a<b ? (c<a?c:a): (c<b?c:b);
}

__device__ __host__ inline int minimum2(const int a,const int b){
  return a<b ? a : b;
}

__device__ __host__ inline int maximum2(const int a,const int b ){
  return a>b ? a : b;
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

void printOverallResults( int argc, char * proc, slownik_entry * slownik, double totalTime, dim3 results[MAX_ARG], char * arguments[MAX_ARG] )
{
  if (env_porcelain)
    printf("TOTAL=%s=%d=%2.6f\n", proc, argc-2, totalTime);
  else
    printf("CZAS CAŁKOWITY (%s): %2.6f\n", proc, totalTime);

  for(int argNum = 2; argNum < argc; argNum++)
    {
      if (env_porcelain)
        {
          printf("%s;%s;%s;%d\n",
                 proc,
                 to_UTF_8(arguments[argNum]),
                 to_UTF_8(slownik[results[argNum].x].slowo),
                 results[argNum].y // odległość
                 );
        }
      else
        {
          printf("%4d. %20s -> %20s : %4d\n",
                 argNum-1,
                 to_UTF_8(arguments[argNum]),
                 to_UTF_8(slownik[results[argNum].x].slowo),
                 results[argNum].y);
        }
    }
}

/////////////// CPU ///////////////
__device__ __host__ inline
int OE_CPU(const char *a,const int aN,
           const char *b,const int bN)
{
  int gt1C[MAX_L];
  int gt2C[MAX_L];

  int *d1 = gt1C;
  int *d2 = gt2C;

  for (int j=0;j<=bN;j++) d1[j]=j;

  for (int i=1;i<=aN;i++) {
    d2[0] = i;
    for (int j=1;j<=bN;j++) {
      d2[j] = minimum3(d1[j  ] + 1,                        // deletion
                       d2[j-1] + 1,                        // insertion
                       d1[j-1] + ((a[i-1]==b[j-1])? 0:1)); // substitution
    }
    d1 = (d1==gt1C)? gt2C:gt1C; // table exchange 1<>2
    d2 = (d2==gt2C)? gt1C:gt2C; // table exchange 1<>2
  }

  return d1[bN];
}

__device__ inline
char OE_GPU(const char * a,const int aN,
            const char * b,const int bN)
{
  char gt1C[MAX_L];
  char gt2C[MAX_L];

  char *d1 = gt1C;
  char *d2 = gt2C;

  for (int j=0;j<=bN;j++) d1[j]=j;

  for (int i=1;i<=aN;i++) {
    d2[0] = i;
    for (int j=1;j<=bN;j++) {
      d2[j] = minimum3(d1[j  ] + 1,                        // deletion
                       d2[j-1] + 1,                        // insertion
                       d1[j-1] + ((a[i-1]==b[j-1])? 0:1)); // substitution
    }
    d1 = (d1==gt1C)? gt2C:gt1C; // table exchange 1<>2
    d2 = (d2==gt2C)? gt1C:gt2C; // table exchange 1<>2
  }

  return d1[bN];
}

void runCPU( char * slowo, slownik_entry * slownik, int rozmiar_slownika,
             int * najblizsze_slowo, int * odleglosc)
{
  char *s = slowo;
  int minlCPU=999999, miniCPU;
  for (int i=0; i<rozmiar_slownika; i++)
    {
      char *d=slownik[i].slowo; //+MAX_L*i;
#if 1
      int   l = OE_CPU(s,  strlen(s),    d,  strlen(d));
#else
      int   l = OE_CPU(d,  strlen(d),    s,  strlen(s));
#endif
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
void kernelGPU_OE(int numer_argumentu, int rozmiar_slownika, int dlugosc_slowa, int * reverse_wyniki)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= rozmiar_slownika )
    {
      return;
    }

  char a[MAX_L];
  char aN = 0; // calculated below

  for(int w=0; w < WORDS_PER_THREAD; w++)
    {
      aN = tex1Dfetch( slownik_tex, sizeof(slownik_entry)*idx+offsetof(slownik_entry,dlugosc));
      for(int i=0; i < aN; i++)
        {
          char c;
          c = tex1Dfetch( slownik_tex, sizeof(slownik_entry)*idx+offsetof(slownik_entry,slowo)+i);
          a[i] = c;
        }

      char val = OE_GPU( a, aN, (const char*)(arguments_gpu_const+numer_argumentu*MAX_L), dlugosc_slowa );

#if 1
      reverse_wyniki[val] = idx;
#else // metoda równie szybka co powyższa, ale dla odmiany zachowuje zgodność z obliczeniami na CPU... zwykle.
      // Czasami nie chce działać, tzn. obliczenia się mimo wszystko rozjeżdżają.
      if ( reverse_wyniki[val] == SPECIAL(rozmiar_slownika) )
        atomicMin(&reverse_wyniki[val], idx);
#endif
    }
}

__host__
void runGPU( int numer_argumentu,
             char * slowo, int rozmiar_slownika,
             int * najblizsze_slowo, int * odleglosc)
{
  int liczba_watkow = align_up(align_up(rozmiar_slownika,TILE),WORDS_PER_THREAD);
  const size_t REVERSE_SIZE = MAX_L+1;

  static int * reverse_wyniki_gpu = NULL;
  if (!reverse_wyniki_gpu)
    cutilSafeCall(cudaMalloc(&reverse_wyniki_gpu, sizeof(int) * REVERSE_SIZE));

  int reverse_wyniki[REVERSE_SIZE];
  for(int i = 0; i < REVERSE_SIZE; i++)
    {
      reverse_wyniki[i] = SPECIAL(rozmiar_slownika);
    }
  cutilSafeCall(cudaMemcpy(reverse_wyniki_gpu, reverse_wyniki, sizeof(int) * REVERSE_SIZE, cudaMemcpyHostToDevice));

  // Wywołanie jądra
  dim3 dimGrid(liczba_watkow/(TILE*WORDS_PER_THREAD));
  dim3 dimBlock(TILE);

#if 0
  printf("liczba_watkow:%d\n", liczba_watkow);
  printf("dimGrid:(%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
  printf("dimBlock:(%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
#endif

  kernelGPU_OE<<<dimGrid, dimBlock>>>(numer_argumentu, rozmiar_slownika, strlen(slowo), reverse_wyniki_gpu);
  cutilSafeCall(cudaMemcpy(reverse_wyniki, reverse_wyniki_gpu, sizeof(int) * REVERSE_SIZE, cudaMemcpyDeviceToHost));

  for(int i = 0; i < REVERSE_SIZE; i++)
    {
      if (reverse_wyniki[i] != SPECIAL(rozmiar_slownika))
        {
          *najblizsze_slowo = reverse_wyniki[i];
          *odleglosc = i;
          return;
        }
    }
}


int main(int argc, char** argv){
  env_porcelain = getenv("PORCELAIN") ? 1 : 0;
#ifdef ALWAYS_PORCELAIN
  env_porcelain = 1;
#endif
  env_skip_cpu = getenv("SKIP_CPU") ? 1 : 0;

  if ( !env_porcelain )
    {
      printf("MAX_L=%d\n", MAX_L);
      printf("MAX_ARG=%d\n", MAX_ARG);
      printf("TILE=%d\n", TILE);
      printf("WORDS_PER_THREAD=%d\n", WORDS_PER_THREAD);
    }

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
#if 0
      printf("Długość słowa: %d\nSłowo: %s\n", dl, buf);
#endif
      max_dl = maximum2( max_dl, dl );
      ilosc++;
    }
  if (!env_porcelain)
    printf("Maksymalna długość: %d\n", max_dl);

  assert( max_dl < MAX_L ); // zakładamy że słowa są krótsze niż MAX_L bajty

  /* wczytujemy słownik faktycznie */
  rewind(plik_slownika);
  size_t slownik_size = align_up(sizeof(slownik_entry) * ilosc, 32);
  slownik_entry * slownik;
  slownik = (slownik_entry *) malloc(slownik_size);
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

      int dl = strlen(buf);
      int common=0;
      char buf2[1024];
      for(int i = 0; i < dl; i++)
        {
          if ( buf[i] != buf2[i] )
            {
              common = i;
              break;
            }
        }
      memcpy(buf2, buf, 1024);
      slownik[cnt].common = cnt ? common : 0;
      slownik[cnt].dlugosc = dl;

      memcpy(slownik[cnt].slowo, buf, MAX_L);
#ifdef DEBUG
      printf("SŁOWO: %s/%d/%d\n", slownik[cnt].slowo, slownik[cnt].dlugosc, slownik[cnt].common);
#endif
      cnt++;
    }
  slownik_rozmiar = cnt;

  if (!env_porcelain)
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
        runCPU( slowo, slownik, env_skip_cpu ? 16 : slownik_rozmiar, &numer_slowa, &odleglosc );
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
        runGPU( argNum, slowo, slownik_rozmiar, &numer_slowa, &odleglosc );
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
