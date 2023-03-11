#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <cuda/std/utility>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <execution>
#include <algorithm>
#include <numeric>
#include <chrono>

int main() {

    // Lets attempt to calculate pi!
    //
    // The way we can do this is to do something called
    // a Monte Carlo method.
    //
    // We will generate points in a square bounded by
    // the points (0,0), (0, 1), (1, 1), (1, 0).
    //
    // We will then see if the points we generate would
    // be in an inscribed circle with radius 1.
    //
    // The ratio between the points in the circle and all
    // points is an approximation of the ratio between the
    // area of the two.
    //
    // The ratio between the two is (pi * 1 ^ 2 / 4) / 1 = pi / 4
    // If we get this ratio then we can multiply by 4 to approximate
    // pi. 

    auto start = std::chrono::high_resolution_clock::now();

    constexpr uint64_t N = 10000000;

    std::vector<uint64_t> in_bounds(N);
    
    std::iota(in_bounds.begin(), in_bounds.end(), 0);

    auto transformation = [] __host__ __device__ (uint64_t n) {
                thrust::minstd_rand rng;
                thrust::random::uniform_real_distribution<float> dist(0.f, 1.f);
                
                // we will discard the first n * 2 numbers since each thread will
                // generate its own number
                rng.discard(n * 2);

                // calling dist(rng) gives us a random number
                float x = dist(rng);
                float y = dist(rng);

                // we are attempting to find all numbers in the radius of 1 from (0,0)
                // so we can use the distance formula
                if(sqrt(x * x + y * y) <= 1) {
                    return 1;  
                }
                return 0;
            };

    std::transform(in_bounds.begin(), in_bounds.end(), in_bounds.begin(), transformation);

    uint64_t found = std::reduce(in_bounds.begin(), in_bounds.end());

    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "estimated PI: " << double(found) / double(N) * 4.0 << " in " << std::chrono::duration<double>(end - start).count() * 1e3 << " ms" << std::endl;
    
    
    return 0;
}
