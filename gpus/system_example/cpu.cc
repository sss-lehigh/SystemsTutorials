#include <iostream>
#include <boost/gil.hpp>
#include <boost/gil/io/read_image.hpp>
#include <boost/gil/io/write_view.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <chrono>

int main(int argc, char** argv) {

    if(argc != 2) {
        std::cout << "Pass in a path to an 8bit rgb png image" << std::endl;
        return 1;
    }

    boost::gil::rgb8_image_t img;

    boost::gil::read_image(argv[1], img, boost::gil::png_tag{});

    float prewitt_filter[]={-1.f,0.f ,1.0f,
                      -1.f,0.f, 1.0f,
                      -1.f,0.f, 1.0f};


    boost::gil::rgb8_image_t filtered(img);

    auto start = std::chrono::high_resolution_clock::now();
    boost::gil::kernel_1d_fixed<float, 9> kernel(prewitt_filter, 4);

    boost::gil::convolve_rows_fixed<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(filtered), kernel, boost::gil::view(filtered));
    boost::gil::convolve_cols_fixed<boost::gil::rgb32f_pixel_t>(boost::gil::const_view(filtered), kernel, boost::gil::view(filtered));
    auto end = std::chrono::high_resolution_clock::now();
    boost::gil::write_view("output.png", boost::gil::view(filtered), boost::gil::png_tag{});

    std::cout << "Created image in " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

    return 0;
}

