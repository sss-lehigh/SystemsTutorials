# Detecting Edges In An Image Image

This is example code demonstrating the power of GPUs.

Run `image_proccess_cpu` and `image_proccess_gpu` with the path to Test.png after
you have built this project. The CPU processing version uses the Boost GIL library
to run its edge detection kernel. The GPU version utilizes CUTLASS, a CUDA library math with
matricies.

Edge detection is done through a mathematical operation called convolution. Convolution
takes an image as a 4-dimensional array and moves a filter (matrix) across the 
image. The filter is directly multiplied with the array and is accumulated into
an output.

If we have a single channel (red, green, or blue) we can write this as:

$O_{h,w} = \sum_{r=0}^{R} \sum_{s=0}^S I_{h+r,w+s} \times W_{R-r-1,S-s-1}$

where O is the output, I is the input, W is our filter, 
R and S are sizes of the filter, and h and w are height and width.

