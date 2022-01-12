#include <iostream>
#include <arrayfire.h>
# include "include/genetic_algorithm.hpp"

af::array consts = af::tile(af::randu(1, 10, 1), 500);


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
    int iters = 1000;
    int dna_size = 10;
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
    //     std::cout << best_score << std::endl;
    // }

    //af_print(consts);

    af::array img = af::constant(0, 900, 900, 4, f32);

    af::array brush1 = af::loadImage("../brushes/1.png", true) / 255.f;
    std::cout << brush1.dims() << std::endl;

    img(af::seq(150), af::seq(300), af::span) = brush1;
    af::array idxs = brush1(af::where(brush1(af::span, af::span, -1) > 0));

    af::array mask = brush1(af::span, af::span, -1) == 0;
    img(af::seq(50, 199), af::seq(100, 399), af::span) = 
        alphaBlend(img(af::seq(50, 199), af::seq(100, 399), af::span), brush1, mask);

    img(af::seq(300, 449), af::seq(300, 599), af::span) = brush1;

    af::Window wnd(800, 800, "Window");
    while (!wnd.close()) wnd.image(img);

    return 0;
}