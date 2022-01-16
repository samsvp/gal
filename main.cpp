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
    Painter::painter("../imgs/example.jpg", "../brushes/3.png");
    for (int i=0; i < 10; i++)
    {
        Painter::run_split_regions();
        if (i >= 5)
            Painter::var_weights = 0;
    }


    af::Window wnd(800, 800, "Preliminary result");
        while (!wnd.close()) wnd.image(Painter::current_img);
    
    af::Window wnd2(Painter::c_weights.dims(0), Painter::c_weights.dims(1), "Weights");
        while (!wnd2.close()) wnd2.image(Painter::c_weights);

    af::Window wnd3(Painter::target_image.dims(0), Painter::target_image.dims(1), "OG");
        while (!wnd3.close()) wnd3.image(Painter::target_image);
    

    af::array mimg = (Painter::current_img * 255).as(u8);
    af::saveImageNative("../imgs/test1.png", mimg);


    int iters = 500;
    int dna_size = 20;
    int pop_size = 500;
    float mutation_rate = 0.01f;
    float cross_amount = 0.5f;

    
    // gfor(af::seq i, 10)
    // {
    //     GeneticAlgorithm gal(pop_size, dna_size, dna_size,
    //     fitness_func, mutation_rate, cross_amount, iters);

    //     gal.run();
    //     af::array best = gal.get_best();
    //     float best_score = gal.get_best_score();
    //     std::cout << best.dims() << std::endl;
    //     //consts(i) = best_score;
        
    //     af_print(consts(0, af::span, af::span) - best);
    //     af_print(consts(0, af::span, af::span));
    //     af_print(best);

    //     std::cout << best.dims() << std::endl;
    //     std::cout << best_score << std::endl;
    //     af_print(best - consts(0, af::span, af::span));
    // }

    //af_print(consts);

    return 0;
}