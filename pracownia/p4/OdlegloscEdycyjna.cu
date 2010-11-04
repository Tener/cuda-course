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

#include <vector>

using namespace std;

iconv_t iconv_from_utf8;
iconv_t iconv_to_utf8;

const int MAX_L=16;  // Maksymalna dlugosc slowa (łącznie z 0 na końcu napisu)
const int MAX_ARG=2048; // Maksymalna liczba argumentów
const int TILE = 16;

__device__ __host__ inline int minimum(const int a,const int b,const int c){
  return a<b? (c<a?c:a): (c<b?c:b);
}


void printResult( char * proc, double time, char * slowo, char * najblizsze, int najblizsze_ix, int odleglosc )
{
  char slowo_utf8[1024];
  char najblizsze_utf8[1024];

  {
    size_t iso_len = strlen(slowo);
    size_t utf_len = 1024;

    char * utf = slowo_utf8;
    char * iso = slowo;

    iconv( iconv_to_utf8,
           &iso, &iso_len,
           &utf, &utf_len);

    * utf = 0;
    * iso = 0;
  }

  {
    size_t iso_len = strlen(najblizsze);
    size_t utf_len = 1024;

    char * utf = najblizsze_utf8;
    char * iso = najblizsze;

    iconv( iconv_to_utf8,
           &iso, &iso_len,
           &utf, &utf_len);

    * utf = 0;
    * iso = 0;
  }


  printf("%s time: %2.6f (ms)\n", proc, time);
  printf("Wzorzec:'%16s' Znaleziony(ix=%03d) :'%16s' (odleglosc=%3d)\n",
         slowo_utf8, najblizsze_ix, najblizsze_utf8, odleglosc);
//  printf("Wzorzec:'%16s' Znaleziony(ix=%03d:'%16s' (odleglosc=%3d)\n",
//         slowo, najblizsze_ix, najblizsze, odleglosc);
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
void kernelGPU_OE(char * slownik, int rozmiar_slownika, char * slowo, int dlugosc_slowa, char * wyniki, int * reverse_wyniki)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= rozmiar_slownika )
    {
      wyniki[idx] = 0;
      return;
    }

  // TODO: use shared memory here instead.
  //  int gt1C[MAX_L];
  //  int gt2C[MAX_L];

  char a[MAX_L]; // słowo ze słownika
  char * b = slowo;

  int aN = 0; // calculated below
  int bN = dlugosc_slowa;

  // loop is good here, can be unrolled at compile time.
  // TODO: consider #pragma unroll(MAX_L)
#pragma unroll 16
  for(int i=0; i < MAX_L; i++)
    {
      if (!slownik[MAX_L*idx+i])
        {
          aN = aN == 0 ? i : aN;
          break; // TODO: check whether is't good to break here
        }
      a[i] = slownik[MAX_L*idx+i];
    }

  //  __syncthreads(); // coalescent memory access
  int val = OE_CPU( a, aN, b, bN );
  wyniki[idx] = val;

  //reverse_wyniki[val] = idx;
  //atomicMin(reverse_wyniki[val], idx);


// 
//  for (int j=0;j<=bN;j++) d1[j]=j;
// 
//  for (int i=1;i<=aN;i++) {
//    d2[0] = i;
//    for (int j=1;j<=bN;j++) {
//      d2[j] = minimum(d1[j  ] + 1,                        // deletion
//                      d2[j-1] + 1,                        // insertion
//                      d1[j-1] + ((a[i-1]==b[j-1])? 0:1)); // substitution
//    }
//    d1 = (d1==gt1C)? gt2C:gt1C; // table exchange 1<>2
//    d2 = (d2==gt2C)? gt1C:gt2C; // table exchange 1<>2
//  }
//  wyniki[idx] = d1[bN];

}

__global__
void kernelGPU_min(char * wyniki, int rozmiar_slownika, dim3 * wynik)
{
  int minOdl = 9999;
  int minIdx;

  for(int i=0; i < rozmiar_slownika; i++)
    {
      if (wyniki[i] < minOdl)
        {
          minOdl = wyniki[i];
          minIdx = i;
        }
    }

   *wynik = dim3( minOdl, minIdx );
}

__host__
void runGPU( char * slowo, char * slownik_gpu, int rozmiar_slownika,
             int * najblizsze_slowo, int * odleglosc)
{
  int liczba_watkow = (1+(rozmiar_slownika / TILE)) * TILE;

  char * slowo_gpu;
  cudaMalloc(&slowo_gpu, MAX_L);
  cudaMemcpy(slowo_gpu, slowo, MAX_L, cudaMemcpyHostToDevice);

  char * wyniki_gpu;
  cudaMalloc(&wyniki_gpu, liczba_watkow);

  // Wywołanie jądra
  dim3 dimGrid(liczba_watkow / TILE);
  dim3 dimBlock(TILE);
  kernelGPU_OE<<<dimGrid, dimBlock>>>(slownik_gpu, rozmiar_slownika, slowo_gpu, strlen(slowo), wyniki_gpu, NULL);

#ifdef DEBUG
  cudaThreadSynchronize();
#endif

  dim3 wynik;
  dim3 * wynik_gpu;
  cudaMalloc(&wynik_gpu, sizeof(dim3));
  kernelGPU_min<<< 1, 1 >>>(wyniki_gpu, rozmiar_slownika, wynik_gpu);
  cudaMemcpy(&wynik,wynik_gpu, sizeof(dim3), cudaMemcpyDeviceToHost);

  *odleglosc = wynik.x;
  *najblizsze_slowo = wynik.y;

#ifdef DEBUG
  char * wyniki = (char *) malloc(liczba_watkow);
  cudaMemcpy(wyniki,wyniki_gpu, liczba_watkow, cudaMemcpyDeviceToHost);
  for(int i=0; i < rozmiar_slownika; i++)
    {
      printf("%02d: %d%s", i, wyniki[i], (i+1) % 4 ? "\t" : "\n" );
    }
  printf("\n");
  free(wyniki);
#endif

#ifdef DEBUG
  cudaFree(slowo_gpu);
  cudaFree(wyniki_gpu);
  cudaFree(wynik_gpu);
  cudaThreadSynchronize();
#endif

}


int main(int argc, char** argv){
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
      fscanf(plik_slownika,"%s",buf);
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
  size_t slownik_size = sizeof(char) * MAX_L * ilosc;
  char * slownik = (char *) malloc(slownik_size);
  int slownik_rozmiar;
  memset(slownik, 0, slownik_size);

  int cnt=0;
  while(!feof(plik_slownika))
    {
      char buf[1024];
      buf[0] = 0;
      fscanf(plik_slownika, "%s", buf);
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


  // --------------------------------------------------------------------
  // -GPU:---------------------------------------------------------------
  // --------------------------------------------------------------------
  /* przesyłanie słownika na GPU */

  char * slownik_GPU;
  cudaMalloc(&slownik_GPU, slownik_size);
  cudaMemcpy(slownik_GPU, slownik, slownik_size, cudaMemcpyHostToDevice);

  for(int argNum = 2; argNum < argc; argNum++)
    {
      char * slowo = arguments[argNum];

      {
        unsigned int timer = 0;
        int numer_slowa = 0, odleglosc = 999;
        // CPU
        cutilCheckError( cutCreateTimer( &timer));
        cutilCheckError( cutStartTimer( timer));
        runCPU( slowo, slownik, slownik_rozmiar, &numer_slowa, &odleglosc );
        printResult("CPU", cutGetTimerValue( timer), slowo, slownik+MAX_L*numer_slowa, numer_slowa, odleglosc);
        cutilCheckError( cutDeleteTimer( timer));
      }

      {
        unsigned int timer = 0;
        int numer_slowa = 0, odleglosc = 999;
        // GPU
        cutilCheckError( cutCreateTimer( &timer));
        cutilCheckError( cutStartTimer( timer));
        runGPU( slowo, slownik_GPU, slownik_rozmiar, &numer_slowa, &odleglosc );
        printResult("GPU", cutGetTimerValue( timer), slowo, slownik+MAX_L*numer_slowa, numer_slowa, odleglosc);
        cutilCheckError( cutDeleteTimer( timer));
      }

    }

  cudaFree(slownik_GPU);
  iconv_close(iconv_from_utf8);
  iconv_close(iconv_to_utf8);

  return 0;


}
