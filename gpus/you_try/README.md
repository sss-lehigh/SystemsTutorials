# You Try

At this point you should have experience looking at C++ and Thrust.

Modify program\_here.cu by translating the standard library C++ calls
to Thrust to gain performance improvements.

In this example you will be approximating $\pi$ using a Monte Carlo method.

The approach creates random points in a square. Some of the points will fall within
a quarter circle. Calculating the ratio of points in the circle to all points 
approximates the ratio of the area of the circle to the area of the square.

This ratio is $\pi / 4$, which means if we multiply the result by 4 we get our approximation.

The following image demonstrates how this works:

![MC Method Image](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)

Notice how increasing points gets a better approximation.

## Output Programs

My solution will be output at the executable `solution` in your build directory 
and can be looked at in the solution folder.

Your code will be output as `your_solution`. Compare the initial runtimes of `your_solution`
and `solution` to see how much performance you can gain by parallelizing this problem on
the GPU.

## Hints

<details>
<summary>
Hint 1
</summary>
<p>
Transformation is already device code. You can execute it on the GPU.
</p>
</details>

<details>
<summary>
Hint 2
</summary>
<p>
You can transform the vectors into thrust host and device vectors.
</p>
</details>


<details>
<summary>
Hint 3
</summary>
<code>std::itoa</code> is similar to <code>thrust::sequence</code> 
</details>

<details>
<summary>
Hint 4
</summary>
<p>
You can combine the transform and reduce into one call or switch them to their thrust counterparts with device vectors
</p>
</details>
