// nvcc -g -lglut -lGLEW  -I$CUDA_SDK/C/common/inc -L$CUDA_SDK/C/lib -lcutil_i386 whitenoise.cu
// Compile ^^^
// ========================================================================
// // Kurs: Procesory graficzne w obliczeniach równoległych 
//          A.Łukaszewski 2010
//=========================================================================
// CUDA-OpenGL interoperability ===========================================
// based on  CUDA SDK : SimpleGL             PBO - pixel buffer object here  
//=========================================================================
#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <curand_kernel.h>

typedef unsigned int  uint;
//=========================================================================
// Pseudo Random: TausStep, LCGStep, Hybrid based on:
// GPU Gems 3:
// Lee Howes, David Thomas (Imperial College London)
// Chapter 37. Efficient Random Number Generation and Application Using CUDA
//=========================================================================
// Cheap pseudo random numbers:
//  
// S1, S2, S3, M - constants,  z - state
__device__ uint TausStep(uint &z, int S1, int S2, int S3, uint M)  {
    uint b=(((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
}

// A, C - constants
__device__ uint LCGStep(uint &z, uint A, uint C) {
    return z=(A*z+C);
}

// Mixed :
__device__ float HybridTaus(uint &z1, uint &z2, uint &z3, uint &z4) {
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    return 2.3283064365387e-10 * (              // Periods
               TausStep(z1, 13, 19, 12, 4294967294UL) ^   // p1=2^31-1
               TausStep(z2,  2, 25,  4, 4294967288UL) ^   // p2=2^30-1
               TausStep(z3,  3, 11, 17, 4294967280UL) ^   // p3=2^28-1
               LCGStep( z4,    1664525, 1013904223UL)     // p4=2^32
           );
}

// Int Mixed and modified: cheaper
__device__ uint HybridTausInt(uint &z1, uint &z2, uint &z3, uint &z4) {
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    return (              // Periods
               TausStep(z1, 13, 19, 12, 4294967294UL) ^   // p1=2^31-1
               //  TausStep(z2,  2, 25,  4, 4294967288UL) ^   // p2=2^30-1
               //  TausStep(z3,  3, 11, 17, 4294967280UL) ^   // p3=2^28-1
               LCGStep( z4,    1664525, 1013904223UL)     // p4=2^32
           );
}

// Testing func:   cheap one int state
__device__ uint funct(uint id) {
    //return LCGStep( id,    1664525, 1013904223UL) ;    // p4=2^32
    return HybridTausInt(id,id,id,id);
    //return id = (1664525*id+1013904223UL) % (65536*256);
    //return id = (xx%256) + 256*(y%256) + 65536*( (256-(xx%256)-(y%256))%256 ) ;
}
//=========================================================================

//=========================================================================
//initialization kernel:
__global__ void initim1(uint * d_output, uint imageW, uint imageH) {
    uint x  = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint y  = (blockIdx.y * blockDim.y) + threadIdx.y;
    uint id = (y * imageW) + x;  // Good for < 16MPix

    if (!( x < imageW && y < imageH ))
      {
        return;
      }
    
    d_output[id] = 0; // 0x424242; // clear screen
}

inline __device__ float3 applyAffine(const float3 old, const float3 params[3])
{
  float3 ret;
  ret.x = params[0].x * old.x + params[0].y * old.y + params[0].z * old.z;
  ret.y = params[1].x * old.x + params[1].y * old.y + params[1].z * old.z;
  ret.z = params[2].x * old.x + params[2].y * old.y + params[2].z * old.z;
  return ret;
}

inline __device__ float3 applyClifford(const float3 old, const float3 params[3])
{
  float3 ret;
  /*
    a = params[0].x;
    c = params[0].y;
    b = params[1].x;
    d = params[1].y;
   */

  ret.x = sinf(params[0].x * old.y) + params[0].y * cosf(params[0].x * old.x);
  ret.y = sinf(params[1].x * old.y) + params[1].y * cosf(params[1].x * old.x);
  ret.z = old.z;
  return ret;

}

//next pseudo random number:
__global__ void modify1(curandState * randStateArr, uint * d_output, const uint imageW, const uint imageH, const uint loop,
                        const float3 aff0[3], const float3 aff1[3], const float3 aff2[3], float threshold
                        ) {
  uint blockId = __umul24(blockIdx.x, blockDim.x) + blockIdx.y;
#ifndef CLIFFORD
  curandState randState = randStateArr[blockId];
#endif 

  uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  uint id = __umul24(y, imageW) + x;  // Good for < 16MPix

  if ((blockIdx.x || blockIdx.y))
    return;

  if (!( x < imageW && y < imageH ))
    {
      return;
    }

  
  float3 vec = { x,y,1 };

  vec = applyAffine(vec, aff0);

  for(int i = 0; i < loop; i++)
    {
      x = rint(vec.x + 30);
      y = rint(vec.y + 30);
      if(   (x > 0)
         && (y > 0) 
         && (x < imageW) 
         && (y < imageH) 
         && (i > 5)
         )
        {
          uint idid = __umul24(y, imageW) + x;  // Good for < 16MPix
          d_output[idid] = 0xDE0000 + i * 0xF;//0xFF; //0xDEADBE;
        }
#ifdef CLIFFORD
      vec = applyClifford( vec, aff1 );
#else
      vec = applyAffine( vec, (curand_uniform(&randState) < threshold) ? aff1 : aff2 );
#endif
    }
  
  // store state back
  //randStateArr[blockId] = randState;
  //__syncthreads();
}

__global__ void randomStateInit(curandState * randStateArr)
{
  uint blockId = __umul24(blockIdx.x, blockDim.x) + blockIdx.y;
  curand_init( 1234 + blockIdx.y, 0xFF & blockIdx.x, 0 , &randStateArr[blockId]);
  __syncthreads();
}



//=========================================================================
// Pseudo Random Kernels END
//=========================================================================


uint width = 573, height = 547;
GLuint   pbo = 0;      // OpenGL PBO id.
uint    *d_output;     // CUDA device pointer to PBO data
curandState * randStateArr; // CURAND state's

dim3 blockSize(16,16); // threads
dim3 gridSize;         // set up in initPixelBuffer

int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer() {
    if (pbo) {      // delete old buffer
        cudaGLUnregisterBufferObject(pbo);
        glDeleteBuffersARB(1, &pbo);
    }

    if (!randStateArr){ // allocate and initialize random generators
      size_t size_randStateArr = sizeof(curandState) * blockSize.x * blockSize.y * blockSize.z;
      cutilSafeCall(cudaMalloc(&randStateArr, size_randStateArr));
      printf("curandStateArr size = %d\n", size_randStateArr);
      dim3 g(1,1,1);
      randomStateInit<<< g, blockSize >>>(randStateArr);
      CUT_CHECK_ERROR("Kernel error");
      cudaThreadSynchronize();
      printf("rand init successful\n");
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    cudaGLRegisterBufferObject(pbo);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // from display:
    cudaGLMapBufferObject((void**)&d_output, pbo  );
    initim1<<<gridSize, blockSize>>>(d_output, width, height);
    CUT_CHECK_ERROR("Kernel error");
    cudaGLUnmapBufferObject(pbo);
}

static int cnt=0; // generation(display calls) count

void display() {
    printf("cnt=%4d w=%4d h=%4d dx/y=%d\n", cnt % 10000, width, height,-1000 + cnt * 100);
    cnt++;


    // przekształcenie bazowe
    float3 aff0[3] = { { 1000, 0, 0/*, -1000 + cnt * 100*/},
                       { 0, 1000, 0/*, -1000 + cnt * 100*/},
                       { 0, 0, 1}};
    
    float threshold = .5;

    // przekształcenia iterowane
#if 0
    float3 aff1[3] = { { -0.4,    0,  -1 },
                       {    0, -0.4, 0.1 },
                       {    0,    0,   1 } };
    
    float3 aff2[3] = { { 0.76, -0.4,   0 },
                       {  0.4, 0.76,   0 },
                       {    0,    0,   1 } };
#endif

#if 0 //DRAGOON
    float3 aff1[3] = { { 0.824074  ,    0.281428, -1.882290 },
                       { -0.212346 , 0.864198, -0.110607 },
                       {    0      ,    0,   1 } };
    
    float3 aff2[3] = { { 0.088272, 0.520988,   0.785360 },
                       { -0.463889, -0.377778, 8.095795 },
                       {    0,    0,   1 } };

    threshold = 0.8;
#endif

#if 0
    float3 aff1[3] = { {    0,    0,   0 },
                       {    0,    0,   0 },
                       {    0,    0,   0 } };
    
    float3 aff2[3] = { {    0,    0,   0 },
                       {    0,    0,   0 },
                       {    0,    0,   0 } };
#endif

#if 1
    float3 aff1[3] = { {  0.5, -0.5,   0 },
                       {  0.5,  0.5,   0 },
                       {    0,    0,   1 } };
    
    float3 aff2[3] = { { -0.5, -0.5,   1 },
                       {  0.5, -0.5,   0 },
                       {    0,    0,   1 } };
#endif


#if CLIFFORD
    float a, b, c, d;
    a = -1.4, b = 1.6, c = 1.0, d = 0.7;

    float3 aff1[3] = { { a, c, 0 },
                       { b, d, 0 },
                       { 0, 0, 1 }};

    float3 aff2[3] = { { 1, 0, 0},
                       { 0, 1, 0},
                       { 0, 0, 1}};
    
    

#endif

    float3 * aff0_dev = NULL;
    float3 * aff1_dev = NULL;
    float3 * aff2_dev = NULL;
    
    // inicjalizacja
    if (!aff0_dev) cudaMalloc(&aff0_dev, sizeof(float3)*3);
    if (!aff1_dev) cudaMalloc(&aff1_dev, sizeof(float3)*3);
    if (!aff2_dev) cudaMalloc(&aff2_dev, sizeof(float3)*3);

    // kopiujemy na urządenie

    cutilSafeCall(cudaMemcpy(aff0_dev, aff0, sizeof(float3)*3, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(aff1_dev, aff1, sizeof(float3)*3, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(aff2_dev, aff2, sizeof(float3)*3, cudaMemcpyHostToDevice));
    
    // wywołanie jądra
    cutilSafeCall(cudaGLMapBufferObject((void**)&d_output, pbo));
    initim1<<<gridSize, blockSize>>>(d_output, width, height);
    CUT_CHECK_ERROR("Kernel error");
    modify1<<<gridSize, blockSize>>>(randStateArr, d_output, width, height, 100,
                                     aff0_dev, aff1_dev, aff2_dev, threshold);
    CUT_CHECK_ERROR("Kernel error");
    cudaGLUnmapBufferObject(pbo );
    cudaThreadSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void keyboard(unsigned char k, int , int ) {
    if (k==27 || k=='q' || k=='Q') exit(1);
    glutPostRedisplay();
}

void reshape(int x, int y) {
    width = x;
    height = y;
    initPixelBuffer();
    glViewport(0, 0, x, y);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup() {
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffersARB(1, &pbo);
}

int main( int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA IFS");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "OpenGL requirements not fulfilled !!!\n");
        exit(-1);
    }
    initPixelBuffer();

    atexit(cleanup);
    glutMainLoop();
    return 0;
}
