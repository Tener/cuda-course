1. Poprawny pomiar czasu wymaga wywołania cudaThreadSynchronize() przed uruchomieniem timera i przed jego zatrzymaniem. Więcej informacji: Cuda best practices, 2.1.1.
2. Koniecznie należy kompilować z optymalizacjami, np. -O3. Nie wpływa to co prawda na kod CUDA, ale wpływa na kod hosta. Przykład:

Bez -O3:

CZAS CAŁKOWITY (CPU): 1162.423950
CZAS CAŁKOWITY (GPU): 222.820007

Z -O3:

CZAS CAŁKOWITY (CPU): 303.368011
CZAS CAŁKOWITY (GPU): 222.748993

3. Każde wywołanie funkcji cuda* powinno być opakowane w kod sprawdzający błędy: cudaSafeCall albo cudaErrorCall.

4. To co się zmieści dajemy do __const__, jak nie to do tekstury i odwołujemy się przez tex1Dfetch, w ostateczności zmienna w __global__.

5. Optymalizacja działa:

>>> a=410.515991+379.635986+379.143005
>>> b=51.917999
>>> a/b
22.521957789628985

a - wersja pierwsza
b - aktualna

6. Przykład uruchomienia:

tener@phenom:~/cuda/p4$ ./a.out slowa.txt fooo bar baz
MAX_L=16
MAX_ARG=16
TILE=256
Maksymalna długość: 14
Wczytano 795875 słów, 795876 nowych linii
CZAS CAŁKOWITY (CPU): 309.966003
   1.                 fooo ->                 foot :    1
   2.                  bar ->                  bar :    0
   3.                  baz ->                  baz :    0
CZAS CAŁKOWITY (GPU): 51.917999
   1.                 fooo ->                 foto :    1
   2.                  bar ->                  bar :    0
   3.                  baz ->                  baz :    0

7. Tester napisany w Haskellu, działa dla: GHC 6.12.3, edit-distance == 0.1.2, split == 0.1.2.1, iconv == 0.4.0.2
   - GHC z paczki dla systemu
   - cabal install edit-distance
   - cabal install split
   - cabal install iconv

8. Zmienna systemowa PORCELAIN sprawia, że wyjście programu jest parsowalne maszynowo.
