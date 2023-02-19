#include <algorithm>
#include <numeric>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <execution>
#include <map>
#include <chrono>

template<typename T>
using vector = thrust::host_vector<T>;

template<typename S, typename T>
using pair = thrust::pair<S, T>;

int main() {

    int size = 1000000;

    // In processing relational data we like to perform agregations, by grouping data by some column and then performing a operation like a sum over each column

    // For example consider we have a coffee buisness with multiple locations broken up by city and we want to find our revenue for each city
    // In SQL and data processing libraries that follow SQL naming this is called a group by

    int num_cities = 10;

    using city_t = unsigned;

    vector<pair<city_t, float>> data;

    for(int i = 0; i < size; ++i) {
        data.push_back({static_cast<unsigned>(rand() % num_cities), rand() / 100.0f});
    }

    // one way we can do this is to perform a map reduce

    auto start = std::chrono::high_resolution_clock::now();

    auto result = std::transform_reduce(data.begin(), data.end(), std::map<city_t, float>{}, [](const std::map<city_t, float>& a, const std::map<city_t, float>& b) {
        
        std::map<city_t, float> m;

        for(auto x : a) {
            m[x.first] = x.second;
        }

        for(auto x : b) {
            m[x.first] += x.second;
        }

        return m;

    }, [](pair<city_t, float> x) {

        std::map<city_t, float> m; 
        m[x.first] = x.second;

        return m;
    });

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;

    // now we parallelize the map reduce
    start = std::chrono::high_resolution_clock::now();

    auto result2 = std::transform_reduce(std::execution::par_unseq, data.begin(), data.end(), std::map<city_t, float>{}, [](const std::map<city_t, float>& a, const std::map<city_t, float>& b) {
        
        std::map<city_t, float> m;

        for(auto x : a) {
            m[x.first] = x.second;
        }

        for(auto x : b) {
            m[x.first] += x.second;
        }

        return m;

    }, [](pair<city_t, float> x) {

        std::map<city_t, float> m; 
        m[x.first] = x.second;

        return m;
    });

    end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;

    // lets compare to thrust
    // CUDA does not have a map type so we must do something else
    
    thrust::device_vector<pair<city_t, float>> data_d(data.size());
    
    thrust::device_vector<city_t> key_d(data.size());
    thrust::device_vector<float> value_d(data.size());
    
    thrust::device_vector<city_t> output_key_d(num_cities);
    thrust::device_vector<float> output_value_d(num_cities);
    
    vector<city_t> output_key(num_cities);
    vector<float> output_value(num_cities);

    start = std::chrono::high_resolution_clock::now();

    thrust::copy(data.begin(), data.end(), data_d.begin());

    // first we sort our data by city
    thrust::sort(data_d.begin(), data_d.end(), [] __host__ __device__ (pair<city_t, float> a, pair<city_t, float> b) {
                return a.first < b.first; 
            });

    // now we unzip the data into keys and values (city as the key, revenue as the value)
    thrust::transform(data_d.begin(), data_d.end(), key_d.begin(), [] __host__ __device__ (pair<city_t, float> a) {
                return a.first;
            });

    thrust::transform(data_d.begin(), data_d.end(), value_d.begin(), [] __host__ __device__ (pair<city_t, float> a) {
                return a.second;
            });

    // now we reduce by key
    // note that reduce by key requires the keys and values to be sorted and will reduce consecutive ranges where the keys are equal
    thrust::reduce_by_key(key_d.begin(), key_d.end(), value_d.begin(), output_key_d.begin(), output_value_d.begin(), [] __host__ __device__(city_t a, city_t b) { return a == b; }, thrust::plus<float>());

    // now we copy back
    thrust::copy(output_key_d.begin(), output_key_d.end(), output_key.begin());
    thrust::copy(output_value_d.begin(), output_value_d.end(), output_value.begin());

    end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration<double>(end - start).count() << std::endl;
}

