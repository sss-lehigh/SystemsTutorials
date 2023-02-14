# Results of Running

When running on a machine with a GTX 1660 Super and Intel Core i7,
we can get the following results.

Note a hand optimized cuda version can perform best, but thrust
is able to achieve close to the performance of hand optimizations
while being significantly more programmable.

```
Duration of parallel cpp version (ms):		8.73581
Duration of parallel tbb version (ms):		9.52444
Duration of parallel thrust version (ms):	1.99218
Duration of parallel cuda version (ms):		1.62102
Duration of sequential version (ms):		52.951
Speedup of best:				            32.6651
```
