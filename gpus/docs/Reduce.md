# Reduce

The reduce pattern is a parallel pattern which takes a list, vector, or other collection 
and runs a binary function with an initial case on each element to reduce to an output.

For example if we have the list `[0 1 2]` we can do a reduce with a function that adds two elements
together and the initial value 0. The result would be `3`. 
To parallelize this we may assign our initial case 0 and `[0]` to thread 1 and `[1 2]` to thread 2.
Thread 1 performs `0 + 0 = 0`, thread 2 performs `1 + 2 = 3`. Then thread 1 may perform `0 + 3 = 3` to
get the result.

## In C++

C++ has a operation in `<algorithm>` which can perform this reduce operation. It is called `std::reduce`.

The most basic version of standard transform below takes an iterator to start at and end at (firstIn and lastIn),
and will iterate performing op on each pair of elements and outputing the result.

```{c++}
template<typename InputIter, typename T, typename Operator>
T std::reduce(InputIter firstIn, InputIter lastIn, T init, Operator op);
```

## Combining with Map

We can futher develop parallel algorithms by combining reduction with [map](Map.md)
to get a parallel pattern called map-reduce. Map-reduce is a well known and
widely used parallel pattern.

To implement this in C++ we can use `std::transform_reduce`.

```{c++}
template<typename InputIter1, InputIter2, typename T, typename OpA, typename OpB>
T std::transform_reduce(InputIter1 start1, InputIter1 end1, InputerIter2 start2, T init, OpA reduce, OpB transform);
```

In this case we assume the containers that we get the iterators from are the same size similar to what we did for map. We perform the transform operation on these inputs and then perform our reduction with reduce to get an output.

Similar to map, we can also use `<execution>` to parallelize the execution of these operations.

We use this pattern in [example2](../example2/) to perform a vector dot product.
