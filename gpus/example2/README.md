# DOT Product

The vector calculation `x dot y` (where x is a vector and y is a vector) is another extremely important calculation. 
It is also used in a variety of settings ranging from:
- solving mathematics for engineering
- simulating problems (such as physics)
- machine learning
- etc.

The problem consists of iterating through the vectors x and y and summing the product.

In order to parallelize this problem on modern hardware, we typically use a [map](../docs/Map.md) followed by the parallel pattern called [reduce](../docs/Reduce.md). 
Reduce will take the output of our code and combine it using some kind of operation (for example plus).

For example when performing:  `[10 11] dot [1 2]` we could map the first index to a thread and the second index to a second thread.
This would lead to thread one calculating `10 * 1` and thread two calculating `11 * 2`. We would then perform a reduction which
would add 10 + 22.

This reduction step can be performed in parallel, although some synchronization between threads will occur to output our results.

For large problems, the GPU will again enable us to parallelize more than the CPU can. 

In this example we demonstrate how to do this on a CPU at a high level and a high level way to do this on the GPU using [Thrust](https://thrust.github.io).
