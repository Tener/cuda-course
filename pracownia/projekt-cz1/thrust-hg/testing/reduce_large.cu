#include <unittest/unittest.h>
#include <thrust/reduce.h>


template <typename T, unsigned int N>
void _TestReduceWithLargeTypes(void)
{
    size_t n = (64 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_data(n);

    for(size_t i = 0; i < h_data.size(); i++)
        h_data[i] = FixedVector<T,N>(i);

    thrust::device_vector< FixedVector<T,N> > d_data = h_data;
    
    FixedVector<T,N> h_result = thrust::reduce(h_data.begin(), h_data.end(), FixedVector<T,N>(0));
    FixedVector<T,N> d_result = thrust::reduce(d_data.begin(), d_data.end(), FixedVector<T,N>(0));

    ASSERT_EQUAL_QUIET(h_result, d_result);
}

void TestReduceWithLargeTypes(void)
{
    _TestReduceWithLargeTypes<int,    1>();
    _TestReduceWithLargeTypes<int,    2>();
    _TestReduceWithLargeTypes<int,    4>();
    _TestReduceWithLargeTypes<int,    8>();
    _TestReduceWithLargeTypes<int,   16>();
    _TestReduceWithLargeTypes<int,   32>();
    _TestReduceWithLargeTypes<int,   64>();
    _TestReduceWithLargeTypes<int,  128>(); 
    _TestReduceWithLargeTypes<int,  256>(); 
    _TestReduceWithLargeTypes<int,  512>();
    //_TestReduceWithLargeTypes<int, 1024>(); // out of memory
    //_TestReduceWithLargeTypes<int, 2048>(); // uses too much local data
    //_TestReduceWithLargeTypes<int, 4096>(); // uses too much local data
}
DECLARE_UNITTEST(TestReduceWithLargeTypes);

template <typename T>
struct plus_mod3
{
    T * table;

    plus_mod3(T * table) : table(table) {}

    __host__ __device__
    T operator()(T a, T b)
    {
        return table[(int) (a + b)];
    }
};

