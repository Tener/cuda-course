#include <unittest/unittest.h>
#include <thrust/gather.h>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>


__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN


template <class Vector>
void TestGatherSimple(void)
{
    typedef typename Vector::value_type T;

    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather(map.begin(), map.end(), src.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 6);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 1);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 2);
}
DECLARE_VECTOR_UNITTEST(TestGatherSimple);


void TestGatherFromDeviceToHost(void)
{
    // source vector
    thrust::device_vector<int> d_src(8);
    d_src[0] = 0; d_src[1] = 1; d_src[2] = 2; d_src[3] = 3; d_src[4] = 4; d_src[5] = 5; d_src[6] = 6; d_src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::host_vector<int> h_dst(5, (int) 0);

    // with map on the device
    thrust::gather(d_map.begin(), d_map.end(), d_src.begin(), h_dst.begin());

    ASSERT_EQUAL(6, h_dst[0]);
    ASSERT_EQUAL(2, h_dst[1]);
    ASSERT_EQUAL(1, h_dst[2]);
    ASSERT_EQUAL(7, h_dst[3]);
    ASSERT_EQUAL(2, h_dst[4]);
}
DECLARE_UNITTEST(TestGatherFromDeviceToHost);


void TestGatherFromHostToDevice(void)
{
    // source vector
    thrust::host_vector<int> h_src(8);
    h_src[0] = 0; h_src[1] = 1; h_src[2] = 2; h_src[3] = 3; h_src[4] = 4; h_src[5] = 5; h_src[6] = 6; h_src[7] = 7;

    // gather indices
    thrust::host_vector<int>   h_map(5); 
    h_map[0] = 6; h_map[1] = 2; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;
    thrust::device_vector<int> d_map = h_map;
   
    // destination vector
    thrust::device_vector<int> d_dst(5, (int) 0);

    // with map on the host
    thrust::gather(h_map.begin(), h_map.end(), h_src.begin(), d_dst.begin());

    ASSERT_EQUAL(6, d_dst[0]);
    ASSERT_EQUAL(2, d_dst[1]);
    ASSERT_EQUAL(1, d_dst[2]);
    ASSERT_EQUAL(7, d_dst[3]);
    ASSERT_EQUAL(2, d_dst[4]);
}
DECLARE_UNITTEST(TestGatherFromHostToDevice);


template <typename T>
void TestGather(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());
    thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGather);


template <class Vector>
void TestGatherIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector flg(5);  // predicate array
    Vector map(5);  // gather indices
    Vector src(8);  // source vector
    Vector dst(5);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 2; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4; src[5] = 5; src[6] = 6; src[7] = 7;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0;

    thrust::gather_if(map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 0);
    ASSERT_EQUAL(dst[3], 7);
    ASSERT_EQUAL(dst[4], 0);
}
DECLARE_VECTOR_UNITTEST(TestGatherIfSimple);

template <typename T>
struct is_even_gather_if
{
    __host__ __device__
    bool operator()(const T i) const
    { 
        return (i % 2) == 0;
    }
};

template <typename T>
void TestGatherIf(const size_t n)
{
    const size_t source_size = std::min((size_t) 10, 2 * n);

    // source vectors to gather from
    thrust::host_vector<T>   h_source = unittest::random_samples<T>(source_size);
    thrust::device_vector<T> d_source = h_source;
  
    // gather indices
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] = h_map[i] % source_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;
    
    // gather stencil
    thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_stencil[i] = h_stencil[i] % 2;
    
    thrust::device_vector<unsigned int> d_stencil = h_stencil;

    // gather destination
    thrust::host_vector<T>   h_output(n);
    thrust::device_vector<T> d_output(n);

    thrust::gather_if(h_map.begin(), h_map.begin(), h_stencil.begin(), h_source.begin(), h_output.begin(), is_even_gather_if<unsigned int>());
    thrust::gather_if(d_map.begin(), d_map.begin(), d_stencil.begin(), d_source.begin(), d_output.begin(), is_even_gather_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIf);


template <typename Vector>
void TestGatherCountingIterator(void)
{

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC) && (_DEBUG != 0)
    KNOWN_FAILURE;
#endif

    typedef typename Vector::value_type T;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(map.begin(),
                   map.end(),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)source.size()),
                   source.begin(),
                   output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::gather(thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((int)output.size()),
                   thrust::make_counting_iterator(0),
                   output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestGatherCountingIterator);

__THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
