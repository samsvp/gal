#pragma once

#include <random>
#include <chrono>
#include <iostream>
#include <arrayfire.h>
#include "genetic_algorithm.hpp"


class Painter : public Score
{
public:
    float var_weights = 1.0f;

    Painter(const char *img_path, const char *brush_path,
        int iters, int dna_size_x, int dna_size_y,
        int loops=10, int pop_size=100, float var_weights=1.0f);
    /*
     * Compares the chosen points color with their respectives color on
     * the target image
     */
    af::array fitness_func(af::array coords) override;
    /*
     * Paints the image using parts of the it and then merges it
     * together
     */
    void run();
    void test_image();

    af::array get_target_img();
    af::array get_current_img();
    af::array get_current_weights();

private:
    int loops;
    int iters;
    int dna_size_x;
    int dna_size_y;
    int pop_size;
    const float PI = 3.14159;
    af::array target_split;

    af::array target_image;
    af::array brush;
    af::array results;
    af::array c_weights;
    af::array current_img;
    af::array img_gradient;

    std::mt19937_64 rng;    
    // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif;

    /*
     * Adds two images using the alpha channel
     */
    af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask);
    /*
     * Metainfo is a Nx4 array containing:
     * (x,y,color,angle) all within the range of 0-1
     */
    af::array make_image(af::array metainfo, bool rotate=false);    
    af::array make_image(af::array metainfo, af::array img,
        bool rotate=false);

    /*
     * Calculates which parts of the image the genetic
     * algorithm should focus on
     */
    af::array calculate_weights(af::array c_img);
};


Painter::Painter(const char *img_path, const char *brush_path,
    int iters, int dna_size_x, int dna_size_y,
    int loops, int pop_size, float var_weights) : 
        pop_size(pop_size), var_weights(var_weights),
        iters(iters), dna_size_x(dna_size_x), 
        dna_size_y(dna_size_y), loops(loops)
{
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().
        time_since_epoch().count();
    std::seed_seq ss {
        uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)
    };
    rng.seed(ss);

    // downsample
    target_image = af::loadImage(img_path) / 255.f;
    target_image = af::medfilt2(target_image, 5, 5);

    // image edges gradient
    af::array dx;
    af::array dy;
    af::sobel(dx, dy, target_image);
    img_gradient = af::abs(af::atan2(dy, dx) / PI / 2);

    // load brush image
    brush = af::loadImage("../brushes/1.png", true) / 255.f;
    brush(af::span, af::span, af::seq(3)) += 0.f;
    brush = af::medfilt2(brush, 5, 5);
    
    brush = af::resize(0.2f, brush);
    std::cout << "brush dims " << brush.dims() << std::endl;
    std::cout << "target image " << target_image.dims() << std::endl;

    unif = std::uniform_real_distribution<double>(0, 1);

    // weights calc
    current_img = af::constant(0, target_image.dims(0), 
        target_image.dims(1), 4, 1, f32);
    c_weights = calculate_weights(current_img);
}


af::array Painter::alpha_blend(const af::array &foreground, 
    const af::array &background, const af::array &mask)
{
    af::array tiled_mask;
    if (mask.dims(2) != foreground.dims(2))
        tiled_mask = tile(mask, 1, 1, foreground.dims(2));
    return foreground * tiled_mask + (1.0f-tiled_mask) * background;
}


af::array Painter::make_image(af::array metainfo, bool rotate)
{
    int img_size_x = target_image.dims(0);
    int img_size_y =  target_image.dims(1);
    af::array img = af::constant(0, img_size_x, img_size_y, 4, 1, f32);
    return make_image(metainfo, img, rotate);
}


af::array Painter::make_image(af::array metainfo, 
    af::array img, bool rotate)
{
    int img_size_x = target_image.dims(0);
    int img_size_y =  target_image.dims(1);

    for (int n=0; n<metainfo.dims(0); n++)
    { 
        af::array mbrush = brush;

        if (rotate)
        {
            mbrush = af::rotate(brush,
                af::sum<float>(metainfo(n, 2)), 0, 
                AF_INTERP_BICUBIC);
        }
        
        int size_x = mbrush.dims(0);
        int size_y = mbrush.dims(1);

        af::array x = af::seq(size_x) + 
            af::tile(metainfo(n, 0) * img_size_x, size_x);
        af::array y = af::seq(size_y) + 
            af::tile(metainfo(n, 1) * img_size_y, size_y);

        af::array mid_x = x(size_x/2);
        af::array mid_y = y(size_x/2);

        // apply color
        mbrush(af::span, af::span, af::seq(3), af::span) = 
            //af::tile(metainfo(n, 2), size_x, size_y, 3);
            af::tile(target_image(mid_x, mid_y, 0), size_x, size_y, 3);

        af::array mask = mbrush(af::span, af::span, -1);
        // af::array mask = mbrush(af::span, af::span, -1);

        img(x, y, af::span, af::span) = 
            alpha_blend(mbrush, img(x, y, af::span, af::span), mask);
    }
    
    return img;
}


af::array Painter::fitness_func(af::array coords)
{
    // coords is pop_size x dna_size_x x 4 x 1
    af::array x = coords(af::span, af::span, 0) * target_image.dims(0);
    af::array y = coords(af::span, af::span, 1) * target_image.dims(1);

    af::array grad = coords(af::span, af::span, 2);

    af::array results = af::constant(0, coords.dims(0), coords.dims(1));

    // we could add the previous inputs into the cost function
    // adding into the stdev part, so the colors will gradually
    // have more weight
    for(int i=0; i<pop_size; i++)
    {
        // Add weight to each loss
        // after a few iterations we shouldn't worry too much
        // about putting points close to each other
        af::array content_loss = af::abs(
            // 1 / weights because we want -max (optimizing towards)
            // the minimum
            af::approx2(1/(c_weights + 1), x(i, af::span), y(i, af::span)) *
            (af::approx2(img_gradient, x(i, af::span), y(i, af::span)) - 
            grad(i, af::span)));

        af::array variance_loss = // we don't want the same inputs 
            var_weights * .1f * af::tile(1/af::stdev(x(i, af::span), AF_VARIANCE_DEFAULT, 1) + 
            1/af::stdev(y(i, af::span), AF_VARIANCE_DEFAULT, 1), 1, coords.dims(1));

        results(i, af::span) = content_loss + variance_loss;        
    }

    return af::sum(
        -(af::pow(results, 2)
        // 1/(af::pow(target_image - paint_img, 2)+0.1f
    ), 1); // sum every row
}


void Painter::run()
{
    float og_weights = var_weights;
    float mutation_rate = 0.001f;
    float cross_amount = 0.5f;

    for (int i=0; i<loops; i++)
    {
        GeneticAlgorithm gal(pop_size, dna_size_x, dna_size_y,
        mutation_rate, cross_amount, iters);

        gal.run(*this);
        af::array best = gal.get_best();

        best = af::reorder(best, 1, 2, 0);

        // instead of just painting over the image
        // we should only paint parts with lower losses
        af::array img = make_image(best, current_img, true);
        c_weights = calculate_weights(img);
        current_img = img;

        // adjust brush size for fine tunning
        if (i == loops / 2)
            brush = af::resize(0.5f, brush);
        if (i == 3 * loops / 4)
            brush = af::resize(0.5f, brush);
    }

    var_weights = og_weights;
}


void Painter::test_image()
{
    float f[] = {0.5, 0.45, 0.5, 0.5, 0.5, 0.5, 1, 0.};
    af::array a(2, 4, f);

    af_print(a);
    af::array img = make_image(a);

    af::Window wnd(800, 800, "Window");
        while (!wnd.close()) wnd.image(img);

    af::array _a = af::randu(2, 4, 1, 20);
    std::cout << "a dims: " << a.dims() << std::endl;
    std::cout << "_a dims: " << _a.dims() << std::endl;
    af::array _img = make_image(_a);

    af::Window wnd2(800, 800, "Window");
        while (!wnd2.close()) wnd2.image(img);
}


af::array Painter::calculate_weights(af::array c_img)
{
    if (c_img.dims(2) != target_image.dims(2) && 
        target_image.dims(2) == 1)
    {
        c_img = c_img(af::span, af::span, 0);
    }

    af::array black = af::constant(0, c_img.dims());
    af::array mask = af::ceil(black + c_img);

    // set weights of unpainted areas as 1
    // weights of painted areas are the difference between
    // the current image and target
    af::array weights = (mask) * (c_img - target_image) + (1 - mask);
    return 3 * af::pow(weights, 2);
}


af::array Painter::get_target_img()
{
    return target_image;
}

af::array Painter::get_current_img()
{
    return current_img;
}

af::array Painter::get_current_weights()
{
    return c_weights;
}