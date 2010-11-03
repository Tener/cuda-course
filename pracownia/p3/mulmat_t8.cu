
#define TILE 8


// Jądro z podziałem na bloki dla większych macierzy
__global__ void MulMatrixKernel2_T8(
    float *A, float *B, float *C, int N)
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

// Jądro z użyciem pamięci dzielonej dla zredukowania dostępów do
// pamieci globalnej
__global__ void MulMatrixKernel3_T8(float* A, float* B, float* C, int N)   // jądro - kernel
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

extern "C" void GPUMatrixMul_T8 (float *A, float *B, float *C, int N) {
    int size = N*N*sizeof(float);
    float  *Ad, *Bd, *Cd;   //  macierze na GPU

    cudaMalloc(&Ad, size);
    cudaMemcpy(Ad, A, size,cudaMemcpyHostToDevice);
    cudaMalloc(&Bd, size);
    cudaMemcpy(Bd, B, size,cudaMemcpyHostToDevice);
    cudaMalloc(&Cd, size);

    // Wywołanie jądra   np.:
    dim3 dimGrid(N/TILE,N/TILE);
    dim3 dimBlock(TILE,TILE);
    MulMatrixKernel2_T8<<<dimGrid, dimBlock>>>(Ad,Bd,Cd,N);

    cudaMemcpy(Cd, C, size,cudaMemcpyDeviceToHost);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}
