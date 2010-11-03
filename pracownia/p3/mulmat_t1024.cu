
#define TILE 1024
#include "mulmat.cu"

extern "C" void GPUMatrixMul_1024 (float *A, float *B, float *C, int N){ GPUMatrixMul(A,B,C,N); }
