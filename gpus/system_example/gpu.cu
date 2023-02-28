#include <iostream>
#include <boost/gil.hpp>
#include <boost/gil/io/read_image.hpp>
#include <boost/gil/io/write_view.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <chrono>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/conv/kernel/default_depthwise_fprop.h>

void convolve(ptrdiff_t height, ptrdiff_t width, std::vector<float>& floatrep) {

    float gaussian[]={0.00022923296f, 0.0059770769f,0.060597949f,
                      0.24173197f,0.38292751f, 0.24173197f,
                      0.060597949f,0.0059770769f,0.00022923296f};

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        float,
        1,
        float,
        float>;
        
    using FpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        float, 
        cutlass::layout::TensorNHWC,
        float, 
        cutlass::layout::TensorNHWC,
        float, 
        cutlass::layout::TensorNHWC,
        float, // element accum
        cutlass::arch::OpClassSimt,        // operator class
        cutlass::arch::Sm50,        // arch tag
        cutlass::gemm::GemmShape<64, 64, 8>,        // tb shape
        cutlass::gemm::GemmShape<32, 32, 8>,       // warp shape
        cutlass::gemm::GemmShape<1, 1, 1>,        // instruction shape
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // threadblock swizzle
        2,                                  // stages
        cutlass::arch::OpMultiplyAdd       // math op tag
        >::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<FpropKernel>;

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    // NHWC
    cutlass::Tensor4DCoord input_size(1l, height, width, 1l);
    cutlass::Tensor4DCoord filter_size(1, 3, 3, 1);
    cutlass::Tensor4DCoord padding(0, 0, 0, 0);
    cutlass::MatrixCoord conv_stride(1, 1);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(1l, height, width, 1l);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1    
    );

    float* a;
    float* w;
    float* out;

    if(cudaMalloc(&a, sizeof(float) * floatrep.size()) != cudaSuccess) {
        exit(1);
    }

    if(cudaMalloc(&out, sizeof(float) * floatrep.size()) != cudaSuccess)
        exit(1);

    if(cudaMalloc(&w, sizeof(float) * 9) != cudaSuccess)
        exit(1);

    if(cudaMemcpy(a, floatrep.data(), sizeof(float) * floatrep.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        exit(1);
    
    if(cudaMemcpy(w, gaussian, sizeof(float) * 9, cudaMemcpyHostToDevice) != cudaSuccess)
        exit(1);
    
    // layout is stride between width, stride between height, stride between N

    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> A(a, cutlass::layout::TensorNHWC(1, width, width * height));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> W(w, cutlass::layout::TensorNHWC(1, 3, 9));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC> Out(out, cutlass::layout::TensorNHWC(1, width, width * height));

    typename EpilogueOp::Params ep{1.0f, 0.0f};
    typename ImplicitGemm::Arguments args(problem_size,
        A,
        W,
        Out,
        Out,
        ep);


    ImplicitGemm op;

    size_t size = op.get_workspace_size(args);

    uint8_t* workspace = nullptr;

    if(size != 0)
        if(cudaMalloc(&workspace, size) != cudaSuccess)
            exit(1);
    
    if(op.can_implement(args) != cutlass::Status::kSuccess) {
        std::cerr << "Unable to implement on gpu" << std::endl;
        exit(1);
    }

    if(op.initialize(args, workspace) != cutlass::Status::kSuccess) {
        std::cerr << "Unable to implement on gpu" << std::endl;
        exit(1);
    }

    if(op() != cutlass::Status::kSuccess) {
        std::cerr << "Didnt work on gpu" << std::endl;
        exit(1);
    }
   
    if(cudaMemcpy(floatrep.data(), out, sizeof(float) * floatrep.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Unable to copy back" << std::endl;
        exit(1);
    }

    cudaFree(a);
    cudaFree(out);
    cudaFree(w);
    cudaFree(workspace);

}

int main(int argc, char** argv) {

    if(argc != 2) {
        std::cout << "Pass in a path to an 8bit rgb png image" << std::endl;
        return 1;
    }

    boost::gil::rgb8_image_t img;

    boost::gil::read_image(argv[1], img, boost::gil::png_tag{});

        

    // WHC tensor
    std::vector<float> floatrepR;
    std::vector<float> floatrepG;
    std::vector<float> floatrepB;

    boost::gil::for_each_pixel(const_view(img), [&](boost::gil::rgb8_pixel_t p) {
                floatrepR.push_back(boost::gil::at_c<0>(p));
                floatrepG.push_back(boost::gil::at_c<1>(p));
                floatrepB.push_back(boost::gil::at_c<2>(p));
            });

    const ptrdiff_t width = img.width();
    const ptrdiff_t height = img.height();

    auto start = std::chrono::high_resolution_clock::now();
    

    
    boost::gil::rgb8_image_t filtered(img.width(), img.height());
   
    convolve(height, width, floatrepR);
    convolve(height, width, floatrepG);
    convolve(height, width, floatrepB);

    int count = 0;

    boost::gil::for_each_pixel(boost::gil::view(filtered), [&](boost::gil::rgb8_pixel_t& p) {
        boost::gil::at_c<0>(p) = static_cast<uint8_t>(floatrepR[count]);
        boost::gil::at_c<1>(p) = static_cast<uint8_t>(floatrepG[count]);
        boost::gil::at_c<2>(p) = static_cast<uint8_t>(floatrepB[count]);
        count++;
    });
    
    auto end = std::chrono::high_resolution_clock::now();
    boost::gil::write_view("output.png", boost::gil::view(filtered), boost::gil::png_tag{});

    std::cout << "Created image in " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    return 0;
}

