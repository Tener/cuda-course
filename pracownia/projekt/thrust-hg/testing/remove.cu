#include <unittest/unittest.h>
#include <thrust/remove.h>
#include <stdexcept>

template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x) { return (static_cast<unsigned int>(x) & 1) == 0; }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(T x) { return x ? true : false; }
};

template<typename Vector>
void TestRemoveSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    typename Vector::iterator end = thrust::remove(data.begin(), 
                                                    data.end(), 
                                                    (T) 2);

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveSimple);


template<typename Vector>
void TestRemoveCopySimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy(data.begin(), 
                                                         data.end(), 
                                                         result.begin(), 
                                                         (T) 2);

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopySimple);


template<typename Vector>
void TestRemoveIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    typename Vector::iterator end = thrust::remove_if(data.begin(), 
                                                      data.end(), 
                                                      is_even<T>());

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveIfSimple);


template<typename Vector>
void TestRemoveIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    typename Vector::iterator end = thrust::remove_if(data.begin(), 
                                                      data.end(),
                                                      stencil.begin(),
                                                      thrust::identity<T>());

    ASSERT_EQUAL(end - data.begin(), 3);

    ASSERT_EQUAL(data[0], 1);
    ASSERT_EQUAL(data[1], 1);
    ASSERT_EQUAL(data[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveIfStencilSimple);


template<typename Vector>
void TestRemoveCopyIfSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(data.begin(), 
                                                           data.end(), 
                                                           result.begin(), 
                                                           is_even<T>());

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopyIfSimple);


template<typename Vector>
void TestRemoveCopyIfStencilSimple(void)
{
    typedef typename Vector::value_type T;

    Vector data(5);
    data[0] =  1; 
    data[1] =  2; 
    data[2] =  1;
    data[3] =  3; 
    data[4] =  2; 

    Vector stencil(5);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 0;
    stencil[3] = 0;
    stencil[4] = 1;

    Vector result(5);

    typename Vector::iterator end = thrust::remove_copy_if(data.begin(), 
                                                           data.end(), 
                                                           stencil.begin(),
                                                           result.begin(), 
                                                           thrust::identity<T>());

    ASSERT_EQUAL(end - result.begin(), 3);

    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 3);
}
DECLARE_VECTOR_UNITTEST(TestRemoveCopyIfStencilSimple);


template<typename T>
void TestRemove(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t h_size = thrust::remove(h_data.begin(), h_data.end(), T(0)) - h_data.begin();
    size_t d_size = thrust::remove(d_data.begin(), d_data.end(), T(0)) - d_data.begin();
    
    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemove);


template<typename T>
void TestRemoveIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), is_true<T>()) - h_data.begin();
    size_t d_size = thrust::remove_if(d_data.begin(), d_data.end(), is_true<T>()) - d_data.begin();
   
    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIf);


template<typename T>
void TestRemoveIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_stencil = h_stencil;
    
    size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<T>()) - h_data.begin();
    size_t d_size = thrust::remove_if(d_data.begin(), d_data.end(), d_stencil.begin(), is_true<T>()) - d_data.begin();
   
    ASSERT_EQUAL(h_size, d_size);

    h_data.resize(h_size);
    d_data.resize(d_size);

    ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveIfStencil);


template<typename T>
void TestRemoveCopy(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), T(0)) - h_result.begin();
    size_t d_size = thrust::remove_copy(d_data.begin(), d_data.end(), d_result.begin(), T(0)) - d_result.begin();
    
    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopy);


template<typename T>
void TestRemoveCopyIf(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<T>()) - h_result.begin();
    size_t d_size = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_true<T>()) - d_result.begin();
    
    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIf);


template<typename T>
void TestRemoveCopyIfStencil(const size_t n)
{
    thrust::host_vector<T>   h_data = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;
    
    thrust::host_vector<bool>   h_stencil = unittest::random_integers<bool>(n);
    thrust::device_vector<bool> d_stencil = h_stencil;
    
    thrust::host_vector<T>   h_result(n);
    thrust::device_vector<T> d_result(n);

    size_t h_size = thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<T>()) - h_result.begin();
    size_t d_size = thrust::remove_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<T>()) - d_result.begin();
    
    ASSERT_EQUAL(h_size, d_size);

    h_result.resize(h_size);
    d_result.resize(d_size);

    ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestRemoveCopyIfStencil);

