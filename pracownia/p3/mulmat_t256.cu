
#define TILE 256
#include "mulmat.cu"
extern "C" void GPUMatrixMul_256 (float *A, float *B, float *C, int N){ GPUMatrixMul(A,B,C,N); }
