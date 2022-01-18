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


int main()
{
    int loops = 20;
    int iters = 500;
    int dna_size_x = 1024;
    int dna_size_y = 3;
    int pop_size = 100;
    float var_weights = 1.0f;

    Painter painter("../imgs/example.jpg", "../brushes/3.png",
        iters, dna_size_x, dna_size_y, 
        loops, pop_size, var_weights);

    painter.run();

    auto target_image = painter.get_target_img();
    auto current_img = painter.get_current_img();
    auto c_weights = painter.get_current_weights();

    af::Window wnd(800, 800, "Preliminary result");
        while (!wnd.close()) wnd.image(current_img);
    
    af::Window wnd2(c_weights.dims(0), c_weights.dims(1), "Weights");
        while (!wnd2.close()) wnd2.image(c_weights);

    af::Window wnd3(target_image.dims(0), target_image.dims(1), "OG");
        while (!wnd3.close()) wnd3.image(target_image);
    

    af::array mimg = (current_img * 255).as(u8);
    af::saveImageNative("../imgs/test1.png", mimg);

    return 0;
}