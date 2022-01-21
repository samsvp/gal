#include <iostream>
#include <arrayfire.h>
# include "include/genetic_algorithm.hpp"
# include "include/painter.hpp"

af::array consts = af::tile(af::randu(1, 20, 20), 500);


af::array fitness_func(af::array A)
{
    return af::sum(af::sum(1/(af::pow(consts - A, 2)+0.1f), 1), 2); // sum every row
}


af::array alphaBlend(const af::array &foreground, const af::array &background, const af::array &mask)
{
    af::array tiledMask;
    if (mask.dims(2)!=foreground.dims(2))
        tiledMask = tile(mask,1,1,foreground.dims(2));
    return foreground*tiledMask+(1.0f-tiledMask)*background;
}


int main(int argc, char **argv)
{
    const char* img_path;
    const char* brush_path;

    // parse arguments
    if (argc==3)
    {
        img_path = argv[1];
        brush_path = argv[2];
    }
    else
    {
        img_path = "../imgs/Monalisa-01.jpg";
        brush_path = "../brushes/3.png";
    }

    int loops = 10;
    int iters = 300;
    int dna_size_x = 2048;
    int dna_size_y = 3;
    int pop_size = 100;
    float brush_scale = 0.2f;
    float var_weights = 1.0f;
    bool save_process = 1;

    Painter painter(img_path, brush_path,
        brush_scale, iters, dna_size_x, dna_size_y, 
        loops, pop_size, var_weights, save_process);

    painter.run();

    auto target_image = painter.get_target_img();
    auto current_img = painter.get_current_img();
    auto c_weights = painter.get_current_weights();
    auto fcurrent_img = af::medfilt2(current_img, 5, 5);

    af::Window wnd(800, 800, "Preliminary result");
        while (!wnd.close()) wnd.image(current_img);
    
    af::Window wnd2(c_weights.dims(0), c_weights.dims(1), "Weights");
        while (!wnd2.close()) wnd2.image(c_weights);

    af::Window wnd3(target_image.dims(0), target_image.dims(1), "OG");
        while (!wnd3.close()) wnd3.image(target_image);
    
    af::Window wnd4(800, 800, "fcurrent_img");
        while (!wnd4.close()) wnd4.image(fcurrent_img);

    af::array mimg = (current_img * 255).as(u8);
    af::saveImageNative("../imgs/test1.png", mimg);

    return 0;
}