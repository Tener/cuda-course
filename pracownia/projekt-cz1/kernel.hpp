#include <thrust/experimental/cuda/pinned_allocator.h>

typedef float ElType;
typedef thrust::experimental::cuda::pinned_allocator< ElType > PAlloc;

typedef thrust::device_vector< ElType, PAlloc > DevVec_P;
typedef thrust::host_vector< ElType, PAlloc > HostVec_P;

typedef thrust::device_vector< ElType > DevVec;
typedef thrust::host_vector< ElType > HostVec;
