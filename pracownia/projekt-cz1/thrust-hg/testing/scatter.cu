#include <unittest/unittest.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

template <class Vector>
void TestScatterSimple(void)
{
    typedef typename Vector::value_type T;

    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    thrust::scatter(src.begin(), src.end(), map.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 2);
    ASSERT_EQUAL(dst[2], 4);
    ASSERT_EQUAL(dst[3], 1);
    ASSERT_EQUAL(dst[4], 0);
    ASSERT_EQUAL(dst[5], 0);
    ASSERT_EQUAL(dst[6], 0);
    ASSERT_EQUAL(dst[7], 3);
}
DECLARE_VECTOR_UNITTEST(TestScatterSimple);


void TestScatterFromHostToDevice(void)
{
    // source vector
    thrust::host_vector<int> h_src(5);
    h_src[0] = 0; h_src[1] = 1; h_src[2] = 2; h_src[3] = 3; h_src[4] = 4;

    // scatter indices
    thrust::device_vector<int> d_map(5);
    d_map[0] = 6; d_map[1] = 3; d_map[2] = 1; d_map[3] = 7; d_map[4] = 2;

    // destination vector
    thrust::device_vector<int> d_dst(8, (int) 0);

    // expected result
    thrust::device_vector<int> reference = d_dst;
    reference[1] = 2;
    reference[2] = 4;
    reference[3] = 1;
    reference[7] = 3;

    // with map on the device
    thrust::scatter(h_src.begin(), h_src.end(), d_map.begin(), d_dst.begin());

    ASSERT_EQUAL(reference, d_dst);
}
DECLARE_UNITTEST(TestScatterFromHostToDevice);


void TestScatterFromDeviceToHost(void)
{
    // source vector
    thrust::device_vector<int> d_src(5);
    d_src[0] = 0; d_src[1] = 1; d_src[2] = 2; d_src[3] = 3; d_src[4] = 4;

    // scatter indices
    thrust::host_vector<int> h_map(5);
    h_map[0] = 6; h_map[1] = 3; h_map[2] = 1; h_map[3] = 7; h_map[4] = 2;

    // destination vector
    thrust::host_vector<int> h_dst(8, (int) 0);

    // expected result
    thrust::host_vector<int> reference(h_dst.size());
    reference[1] = 2;
    reference[2] = 4;
    reference[3] = 1;
    reference[7] = 3;

    // with map on the host
    thrust::scatter(d_src.begin(), d_src.end(), h_map.begin(), h_dst.begin());

    ASSERT_EQUAL(reference, h_dst);
}
DECLARE_UNITTEST(TestScatterFromDeviceToHost);


template <typename T>
void TestScatter(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());
    thrust::scatter(d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestScatter);


template <class Vector>
void TestScatterIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector flg(5);  // predicate array
    Vector map(5);  // scatter indices
    Vector src(5);  // source vector
    Vector dst(8);  // destination vector

    flg[0] = 0; flg[1] = 1; flg[2] = 0; flg[3] = 1; flg[4] = 0;
    map[0] = 6; map[1] = 3; map[2] = 1; map[3] = 7; map[4] = 2;
    src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; src[4] = 4;
    dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0; dst[4] = 0; dst[5] = 0; dst[6] = 0; dst[7] = 0;

    thrust::scatter_if(src.begin(), src.end(), map.begin(), flg.begin(), dst.begin());

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 0);
    ASSERT_EQUAL(dst[2], 0);
    ASSERT_EQUAL(dst[3], 1);
    ASSERT_EQUAL(dst[4], 0);
    ASSERT_EQUAL(dst[5], 0);
    ASSERT_EQUAL(dst[6], 0);
    ASSERT_EQUAL(dst[7], 3);
}
DECLARE_VECTOR_UNITTEST(TestScatterIfSimple);


template <typename T>
class is_even_scatter_if
{
    public:
    __host__ __device__ bool operator()(const T i) const { return (i % 2) == 0; }
};

template <typename T>
void TestScatterIf(const size_t n)
{
    const size_t output_size = std::min((size_t) 10, 2 * n);
    
    thrust::host_vector<T> h_input(n, (T) 1);
    thrust::device_vector<T> d_input(n, (T) 1);
   
    thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

    for(size_t i = 0; i < n; i++)
        h_map[i] =  h_map[i] % output_size;
    
    thrust::device_vector<unsigned int> d_map = h_map;

    thrust::host_vector<T>   h_output(output_size, (T) 0);
    thrust::device_vector<T> d_output(output_size, (T) 0);

    thrust::scatter_if(h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), h_output.begin(), is_even_scatter_if<unsigned int>());
    thrust::scatter_if(d_input.begin(), d_input.end(), d_map.begin(), d_map.begin(), d_output.begin(), is_even_scatter_if<unsigned int>());

    ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestScatterIf);


template <typename Vector>
void TestScatterCountingIterator(void)
{
    typedef typename Vector::value_type T;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);

    Vector output(10);

    // source has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                    map.begin(),
                    output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(source.begin(), source.end(),
                    thrust::make_counting_iterator(0),
                    output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                    thrust::make_counting_iterator(0),
                    output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestScatterCountingIterator);


template <typename Vector>
void TestScatterIfCountingIterator(void)
{
    typedef typename Vector::value_type T;

    Vector source(10);
    thrust::sequence(source.begin(), source.end(), 0);

    Vector map(10);
    thrust::sequence(map.begin(), map.end(), 0);
    
    Vector stencil(10, 1);

    Vector output(10);

    // source has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                       map.begin(),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
    
    // map has any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(source.begin(), source.end(),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
    
    // source and map have any_space_tag
    thrust::fill(output.begin(), output.end(), 0);
    thrust::scatter_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(10),
                       thrust::make_counting_iterator(0),
                       stencil.begin(),
                       output.begin());

    ASSERT_EQUAL(output, map);
}
DECLARE_VECTOR_UNITTEST(TestScatterIfCountingIterator);

