# Map

The map pattern is a parallel pattern which takes a list, vector, or other object 
and runs a function on each element to get a list, vector, or other set of objects output.

For example if we have the list `[0 1 2 3]` we can do a map with a function that adds one to each 
element. The result would be `[1 2 3 4]`. To parallelize this we could map each element to a thread
of execution. This would mean the first thread gets 0 and outputs 1, the second thread gets 1 and outputs
2, and so on.

## In C++

C++ has a operation in `<algorithm>` which can perform this map operation. It is called `std::transform`.

The most basic version of standard transform below takes an iterator to start at and end at (firstIn and lastIn),
and will iterate performing op on each element and outputing to the output iterator starting from firstOut. It is
assumed that the size of the output is at least the size of the input.

```{c++}
template<typename InputIter, typename OutputIter, typename Operator>
constexpr OutputIter std::transform(InputIter firstIn, InputIter lastIn, OutputIter firstOut, Operator op);
```

Transform can also take two intputs. The `InputIter2` passed below will be taken as a second input. While the first container
is iterated through from firstIn to lastIn, the second container given by the `InputIter2` will be iterated to. The `Operator op`
will be called on both of the inputs and the result will be output to output iterator. We assume the second container passed and the
output is at least the side of the first input.

```{c++}
template<typename InputIter, typename InputIter2, typename OutputIter, typename Operator>
constexpr OutputIter std::transform(InputIter firstIn, InputIter lastIn, InputIter2 firstIn2, OutputIter firstOut, Operator op);
```

In our scalar a x plus y (SAXPY) implementation given in [example 1](../example1) this can be used to output to y by doing:

```{c++}
std::transform(x.begin(), x.end(), y.begin(), output.begin(), op);
```

Where op multiplies the element of x by a, adds it with the element of y, and stores it in output.

To further extend this we can optionally use `<execution>` to pass a execution policy to the transform or any standard algorithm that
supports it.

## C++ Execution Policies

Execution policies in C++ include:
- `std::seq` which will run the algorithm as sequential
- `std::unseq` which enables the algorithm to be vectorized (using vector instructions)
- `std::par` which will enable running the algorithm in parallel across threads
- `std::par_unseq` which will enable running the algorithm in parallel and vectorizing it

We can use these in the example code above like this:

```{c++}
std::transform(std::par_unseq, x.begin(), x.end(), y.begin(), output.begin(), op);
```

The implementer of the standard C++ library can now parallelize and vectorize our code.

