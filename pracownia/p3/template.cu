
template <int TILE>
class MulMat
{
public:
__global__ void MulMatrixKernel2(float *A, float *B, float *C, int N)
  {
    float res=0;
    int x=blockIdx.x * TILE + threadIdx.x; // zmiany
    int y=blockIdx.y * TILE + threadIdx.y; // zmiany

    for (int k=0; k<N; ++k) {
      float Ai=A[y * N + k];
      float Bi=B[k * N + x];
      res = Ai*Bi;
    }
    C[y * N + x]=res;
  }

__global__ void MulMatrixKernel3(float* A, float* B, float* C, int N)   // jądro - kernel
  {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // wiersz i kolumna elementu macierzy C do obliczenia
    int ypos = by * TILE + ty;
    int xpos = bx * TILE + tx;

    float res = 0;   //  rezultat obliczany

    // petla po kafelkach/tiles macierzy dla tego bloku
    for (int m = 0; m < N/TILE; ++m) {

      // wątki ładują wspólnie po elemencie A i B kafelka do pamieci dzielonej
      As[ty][tx] = A[xpos*N + (m*TILE + tx)];
      Bs[ty][tx] = B[(m*TILE + ty)*N + ypos];
      __syncthreads();

      for (int k = 0; k < TILE; ++k)      // wlaściwe obliczenia
        res += As[ty][k] * Bs[k][tx];
      __syncthreads();
    }
    C[ypos*N + xpos] = res;                 // zwracamy do pamięci globalnej
  }


private:
} ;


int main() {
  float * A;
  float * B;
  float * C;

  MulMat< 8 >.MulMatrixKernel3( A, B, C, 8 );
}
