#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// thrust
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>

#include <thrust/experimental/cuda/pinned_allocator.h>



const int ITER_CNT = 1000;
const int VEC_SIZE = 1024 * 1;


// initialize one vector from another
template <typename VecSrc, typename VecTgt>
void test1(std::string testName)
{

  std::cerr << testName << std::endl;
  
  VecSrc vec(VEC_SIZE);
  //thrust::generate(vec.begin(), vec.end(), rand);
  for(int i = 0; i < vec.size(); i++) vec[i] = rand();

  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
    
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);

  for(int i = 0; i < ITER_CNT; i++)
    {   
      VecTgt tvec(vec);
      std::cout << thrust::reduce( tvec.begin(), tvec.end(), 0 ) << " ";
    }
  
  std::cout << std::endl;

  cudaThreadSynchronize();
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cerr << "Total time: " << elapsed_time << " milliseconds" << std::endl;
}


// initialize one vector from another - but using device/raw_pointer_cast
template <typename VecSrc, typename VecTgt>
void test2_1(std::string testName)
{
  std::cerr << testName << std::endl;
  
  VecSrc vec(VEC_SIZE);
  //thrust::generate(vec.begin(), vec.end(), rand);
  for(int i = 0; i < vec.size(); i++) vec[i] = rand();

  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
    
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);
  //
  for(int i = 0; i < ITER_CNT; i++)
    {   
      VecTgt tvec( thrust::device_pointer_cast(&*vec.begin()), thrust::device_pointer_cast(&*vec.end()) );
      std::cout << thrust::reduce( tvec.begin(), tvec.end(), 0 ) << " ";
    }
  //
  cudaThreadSynchronize();
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cerr << "Total time: " << elapsed_time << " milliseconds" << std::endl;
}

// initialize one vector from another - but using device/raw_pointer_cast
template <typename VecSrc, typename VecTgt>
void test2_2(std::string testName)
{
  std::cerr << testName << std::endl;
  
  VecSrc vec(VEC_SIZE);
  //thrust::generate(vec.begin(), vec.end(), rand);
  for(int i = 0; i < vec.size(); i++) vec[i] = rand();

  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
    
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);
  //
  for(int i = 0; i < ITER_CNT; i++)
    {   
      VecTgt tvec( thrust::raw_pointer_cast(&*vec.begin()), thrust::raw_pointer_cast(&*vec.end()) );
      std::cout << thrust::reduce( tvec.begin(), tvec.end(), 0 ) << " ";
    }
  //
  cudaThreadSynchronize();
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cerr << "Total time: " << elapsed_time << " milliseconds" << std::endl;
}

// fill vector type with thrust::sequence
template <typename VecT>
void test3(std::string testName)
{

  std::cerr << testName << std::endl;
  
  cudaEvent_t start;
  cudaEvent_t end;
  float elapsed_time;
    
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start,0);
  //
  for(int i = 0; i < ITER_CNT; i++)
    {   
      VecT tvec(VEC_SIZE);
      thrust::sequence( tvec.begin(), tvec.end() );
    }
  //
  cudaThreadSynchronize();
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);
  std::cerr << "Total time: " << elapsed_time << " milliseconds" << std::endl;
}

int main(void)
{
  srand(time(0));

  typedef int ElType;
  typedef thrust::experimental::cuda::pinned_allocator< ElType > Alloc;

  std::cerr << "ITER_CNT = " << ITER_CNT << std::endl;
  std::cerr << "VEC_SIZE = " << VEC_SIZE << std::endl; 

  std::cerr << std::endl;

  if ( 1 )
  {
    std::cerr << "test1()" << std::endl;

    test1< thrust::host_vector< ElType >		, thrust::device_vector< ElType > >("normal|normal mem (h,d)");
    test1< thrust::device_vector< ElType >		, thrust::device_vector< ElType > >("normal|normal mem (d,d)");
    test1< thrust::host_vector< ElType >		, thrust::host_vector< ElType > >("normal|normal mem (h,h)");
    test1< thrust::device_vector< ElType >		, thrust::host_vector< ElType > >("normal|normal mem (d,h)");
    std::cerr << std::endl;
 
    test1< thrust::host_vector< ElType, Alloc >		, thrust::device_vector< ElType > >("pinned|normal mem (h,d)");
    test1< thrust::device_vector< ElType, Alloc >	, thrust::device_vector< ElType > >("pinned|normal mem (d,d)");
    test1< thrust::host_vector< ElType, Alloc >		, thrust::host_vector< ElType > >("pinned|normal mem (h,h)");
    test1< thrust::device_vector< ElType, Alloc >	, thrust::host_vector< ElType > >("pinned|normal mem (d,h)");
    std::cerr << std::endl;
 
    test1< thrust::host_vector< ElType >		, thrust::device_vector< ElType, Alloc > >("normal|pinned mem (h,d)");
    test1< thrust::device_vector< ElType >		, thrust::device_vector< ElType, Alloc > >("normal|pinned mem (d,d)");
    test1< thrust::host_vector< ElType >		, thrust::host_vector< ElType, Alloc > >("normal|pinned mem (h,h)");
    test1< thrust::device_vector< ElType >		, thrust::host_vector< ElType, Alloc > >("normal|pinned mem (d,h)");
    std::cerr << std::endl;

    test1< thrust::host_vector< ElType, Alloc >		, thrust::device_vector< ElType, Alloc > >("pinned|pinned mem (h,d)");
    test1< thrust::device_vector< ElType, Alloc >	, thrust::device_vector< ElType, Alloc > >("pinned|pinned mem (d,d)");
    test1< thrust::host_vector< ElType, Alloc >		, thrust::host_vector< ElType, Alloc > >("pinned|pinned mem (h,h)");
    test1< thrust::device_vector< ElType, Alloc >	, thrust::host_vector< ElType, Alloc > >("pinned|pinned mem (d,h)");
    std::cerr << std::endl;
  }

  {
    std::cerr << "test2_1()" << std::endl;

    test2_1< thrust::device_vector< ElType >, thrust::device_vector< ElType > >("normal|normal mem (d,d)");
    test2_1< thrust::device_vector< ElType >, thrust::device_vector< ElType, Alloc > >("normal|pinned mem (d,d)");
    std::cerr << std::endl;
 
    test2_1< thrust::device_vector< ElType >, thrust::host_vector< ElType > >("normal|normal mem (d,h)");
    test2_1< thrust::device_vector< ElType >, thrust::host_vector< ElType, Alloc > >("normal|pinned mem (d,h)");
    std::cerr << std::endl;

    std::cerr << "test2_2()" << std::endl;

    test2_2< thrust::device_vector< ElType, Alloc >, thrust::device_vector< ElType > >("pinned|normal mem (d,d)");
    test2_2< thrust::device_vector< ElType, Alloc >, thrust::device_vector< ElType, Alloc > >("pinned|pinned mem (d,d)");
    std::cerr << std::endl;

    test2_2< thrust::device_vector< ElType, Alloc >, thrust::host_vector< ElType > >("pinned|normal mem (d,h)");
    test2_2< thrust::device_vector< ElType, Alloc >, thrust::host_vector< ElType, Alloc > >("pinned|pinned mem (d,h)");
    std::cerr << std::endl;
  }

  {
    std::cerr << "test3()" << std::endl;

    test3< thrust::host_vector< ElType > >		("normal h");
    test3< thrust::host_vector< ElType, Alloc > >	("pinned h");
    test3< thrust::device_vector< ElType > >		("normal d");
    test3< thrust::device_vector< ElType, Alloc > >	("pinned d");
    std::cerr << std::endl;
  }

  return 0;
}
