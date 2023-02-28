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

