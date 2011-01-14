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
#include <cuda_gl_interop.h>

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
    uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint id = __umul24(y, imageW) + x;  // Good for < 16MPix

    if ( x < imageW && y < imageH ) {
        d_output[id] = id;
    }
}

//next pseudo random number:
__global__ void modify1(uint * d_output, uint imageW, uint imageH) {
    uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint id = __umul24(y, imageW) + x;  // Good for < 16MPix

    if ( x < imageW && y < imageH ) {
        d_output[id] = funct(d_output[id]);
    }
}
//=========================================================================
// Pseudo Random Kernels END
//=========================================================================


uint width = 573, height = 547;
GLuint   pbo = 0;      // OpenGL PBO id.
uint    *d_output;     // CUDA device pointer to PBO data

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
    printf("%4d\n", cnt % 10000);
    cnt++;

    cudaGLMapBufferObject((void**)&d_output, pbo  );
    modify1<<<gridSize, blockSize>>>(d_output, width, height);
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
    glutCreateWindow("CUDA WhiteNoise");
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
