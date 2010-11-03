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

const int MAX_L=16;  // Maksymalna dlugosc slowa (łącznie z 0 na końcu napisu)

//========================================================================
// Funkcja testowa OE_CPU() do porównywania na CPU
//
// Tablice tymczasowe do uzycia w liczeniu odleglosci edycyjnej na CPU
int gt1C[MAX_L];
int gt2C[MAX_L];

inline int minimum(const int a,const int b,const int c){
  return a<b? (c<a?c:a): (c<b?c:b);
}

// Odległosc edycyjna Levenshtein'a
inline int OE_CPU(char *a,const int aN,char *b,const int bN){
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

inline int maximum( int a, int b ){ return a > b ? a : b; }

//========================================================================
// Bardzo prymitywny test funkcji OE_CPU i wzorzec jak i czym mierzyć czas
//
int main(int argc, char** argv){
  setlocale(LC_ALL, "ISO-8859-2");

  if ( argc < 3 )
    {
      printf("Za mało argumentów\n");
      return 1;
    }

  // --------------------------------------------------------------------
  // Zaladuj slownik
  FILE *plik_slownika =         fopen(argv[1], "r");  // pierwszy argument !!!
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
      printf("Długość słowa: %d\nSłowo: %s\n", dl, buf);
      max_dl = maximum( max_dl, dl );
      ilosc++;
    }
  printf("Maksymalna długość: %d\n", max_dl);
  assert( max_dl < MAX_L ); // zakładamy że słowa są krótsze niż MAX_L bajty

  /* wczytujemy słownik faktycznie */
  rewind(plik_slownika);
  size_t slownik_size = sizeof(char) * MAX_L * ilosc;
  char * slownik = (char *) malloc(slownik_size);
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
      printf("SŁOWO: %s\n", slownik + cnt * MAX_L);
      cnt++;
    }

  printf("Wczytano %d słów, %d nowych linii\n", cnt, ilosc);

  /* konwertujemy argumenty */
  iconv_t iconv_from_utf8 = iconv_open("ISO-8859-2","UTF-8");
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

      printf("argv[%d] = %s(%d) = %s(%d)\n",
             i,
             argv[i], strlen(argv[i]),
             buf, strlen(buf));
    }

#ifdef NOTDEF

  // ...   do prostego testu chwilowo wstawiamy tablice ze slowami co 16
  int  nSlow=3;
  char pSlowa[] = "domk\0          \ndommek\0        \ndonek\0         \n";

  // --------------------------------------------------------------------
  // -GPU:---------------------------------------------------------------
  // --------------------------------------------------------------------
  // Slownik możemy przeslać na GPU tutaj: tego nie liczymy do czasu
  // Tylko słownik, nie słowa !
  // ...

  // Testowanie na wszystkich slowach ktore sa kolejnymu argumentami
  // wywolania programu (argv, poza pierwszym: plik slownika) GPU/CPU
  // --------------------------------------------------------------------
  unsigned int timer = 0;
  cutilCheckError( cutCreateTimer( &timer));
  cutilCheckError( cutStartTimer(   timer));

  // GPU load run get
  // ...

  printf( "GPU time: %2.6f (ms)\n", cutGetTimerValue( timer));
  cutilCheckError( cutDeleteTimer( timer));
  // GPU rezultaty wypisać tu:
  // ...

  // --------------------------------------------------------------------
  // -CPU:---------------------------------------------------------------
  // --------------------------------------------------------------------
  cutilCheckError( cutCreateTimer( &timer));
  cutilCheckError( cutStartTimer( timer));

  // CPU run
  // ...
  char *s=argv[2];  // na poczatek tylko pierwsze slowo <- testowac wszystkie
  int minlCPU=999999, miniCPU;
  for (int i=0; i<nSlow; i++)
    {
      char *d=pSlowa+MAX_L*i;
      int   l = OE_CPU(s,  strlen(s),    d,  strlen(d));
      if (l<minlCPU){   minlCPU=l;  miniCPU=i; }
    }
  printf( "CPU time: %2.6f (ms)\n", cutGetTimerValue( timer));
  cutilCheckError( cutDeleteTimer( timer));

  // CPU rezultaty wypisać tu:
  printf("Wzorzec:%16s Znaleziony:%16s (odleglosc=%3d)\n",
         s,      pSlowa+MAX_L*miniCPU, minlCPU);

  // --------------------------------------------------------------------
#endif

  return 0;


}
