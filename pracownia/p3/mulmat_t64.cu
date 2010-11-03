
#define TILE 64
#include "mulmat.cu"
extern "C" void GPUMatrixMul_64 (float *A, float *B, float *C, int N){ GPUMatrixMul(A,B,C,N); }
