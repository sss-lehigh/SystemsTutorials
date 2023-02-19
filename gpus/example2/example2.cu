#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <execution>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cuda/std/functional>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// dot product
// we do sum a_i * b_i

// Using vector type thrust::host_vector<T>
template<typename T>
using vector = thrust::host_vector<T>;

int main() {
    // creating our vectors x and y of size
    const int size = 1000000;

    vector<float> x(size, 0);
    vector<float> y(size, 0);
    
    thrust::device_vector<float> x_device(x.size());
    thrust::device_vector<float> y_device(y.size());


    // random initialization
    for(int i = 0; i < size; ++i) {
        x[i] = rand() / static_cast<float>(RAND_MAX);
        y[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // sequential execution for reference, we do plus as our reduction and times as our map
    auto start = std::chrono::high_resolution_clock::now();
    float reference = std::transform_reduce(std::execution::seq, x.begin(), x.end(), y.begin(), 0.0f, [](float x, float y) {
                return x + y;
            }, [](float x, float y)  {
                return x * y;
            });
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration<double>(end - start).count() * 1e3 << " ms" << std::endl;

    // execute in parallel using the C++ standard library similar to example 1
    start = std::chrono::high_resolution_clock::now();
    float val = std::transform_reduce(std::execution::par_unseq, x.begin(), x.end(), y.begin(), 0.0f, [](float x, float y) {
                return x + y;
            }, [](float x, float y)  {
                return x * y;
            });
    end = std::chrono::high_resolution_clock::now();

    if(std::abs(val - reference) > std::abs(reference * 1e-3)) {
        std::cerr << "Result is off by more than 1e-3 tolerance " << val << " " << reference << std::endl;
    }

    std::cout << std::chrono::duration<double>(end - start).count() * 1e3 << " ms" << std::endl;

    // We create two lambdas and mark them as executable on both the host and the device
    auto times = [] __host__ __device__ (thrust::tuple<float, float> x) -> float {
                        return thrust::get<0>(x) * + thrust::get<1>(x);
                     };

    auto plus = [] __host__ __device__ (float x, float y) -> float {
                        return x + y;
                     };

    
    start = std::chrono::high_resolution_clock::now();

    // copy memory to the GPU
    thrust::copy(x.begin(), x.end(), x_device.begin());
    thrust::copy(y.begin(), y.end(), y_device.begin());

    // create an iterator that zips x and y together, as in element one is {x[0], y[0]},
    // element 2 is {x[1], y[1]}, and so on
    auto begin_iter = thrust::make_zip_iterator(x_device.begin(), y_device.begin());
    auto end_iter = thrust::make_zip_iterator(x_device.end(), y_device.end());

    // perform the transform reduce
    val = thrust::transform_reduce(begin_iter, end_iter, times, 0.0f, plus);
    
    end = std::chrono::high_resolution_clock::now();

    if(std::abs(val - reference) > std::abs(reference * 1e-3)) {
        std::cerr << "Result is off by more than 1e-3 tolerance " << val << " " << reference << std::endl;
    }

    std::cout << std::chrono::duration<double>(end - start).count() * 1e3 << " ms" << std::endl;

    return 0;
}

