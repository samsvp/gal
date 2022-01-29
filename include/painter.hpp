#pragma once

#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <arrayfire.h>

#include "image_functions.hpp"
#include "genetic_algorithm.hpp"


class Painter : public Score
{
public:
    float var_weights = 1.0f;
    float grad_weights = 1.0f;

    Painter(const char *img_path, const char *brush_path,
        float brush_scale, int iters, int dna_size_x, 
        int dna_size_y, int loops=10, int pop_size=100,
        float var_weights=1.0f, float grad_weights=1.0f);
    ~Painter();

    /*
     * Compares the chosen points color with their respectives color on
     * the target image
     */
    const af::array fitness_func(af::array coords) override;
    /*
     * Paints the image using parts of the it and then merges it
     * together
     */
    void run(bool save=false);

    af::array get_target_img() const;
    af::array get_current_img() const;
    af::array get_current_weights() const;

private:
    int loops;
    int iters;
    int dna_size_x;
    int dna_size_y;
    int pop_size;
    af::array target_split;

    af::array target_image;
    af::array brush;
    af::array results;
    af::array c_weights;
    af::array current_img;
    af::array img_gradient;

    /*
     * Metainfo is a Nx4 array containing:
     * (x,y,color,angle) all within the range of 0-1
     */
    af::array make_image(af::array metainfo, 
        bool save=false, int* frame_n=0, bool rotate=false) const;    
    af::array make_image(af::array metainfo, af::array img,
        bool save=false, int* frame_n=0, bool rotate=false) const;

    /*
     * Calculates which parts of the image the genetic
     * algorithm should focus on
     */
    af::array calculate_weights(af::array c_img) const;
};


Painter::Painter(const char *img_path, const char *brush_path,
    float brush_scale, int iters, int dna_size_x,
    int dna_size_y, int loops, int pop_size, 
    float var_weights, float grad_weights) : 
        pop_size(pop_size), var_weights(var_weights),
        iters(iters), dna_size_x(dna_size_x), 
        dna_size_y(dna_size_y), loops(loops),
        grad_weights(grad_weights)
{
    // downsample
    target_image = af::loadImage(img_path, 1) / 255.f;
    target_image = af::medfilt2(target_image, 5, 5);

    // image edges gradient
    af::array target_gray = af::rgb2gray(target_image);
    af::array dx;
    af::array dy;
    af::sobel(dx, dy, target_gray);
    img_gradient = af::abs(af::atan2(dy, dx));

    // load brush image
    brush = af::loadImage(brush_path, true) / 255.f;
    brush(af::span, af::span, af::seq(3)) += 0.f;
    brush = af::medfilt2(brush, 5, 5);
    
    brush = af::resize(brush_scale, brush);
    std::cout << "brush dims " << brush.dims() << std::endl;
    std::cout << "target image " << target_image.dims() << std::endl;

    // weights calc
    current_img = af::constant(0, target_image.dims(0), 
        target_image.dims(1), 4, 1, f32);
    c_weights = calculate_weights(current_img);
}


Painter::~Painter()
{
    
}


af::array Painter::make_image(af::array metainfo, 
    bool save, int* frame_n, bool rotate) const
{
    int img_size_x = target_image.dims(0);
    int img_size_y =  target_image.dims(1);
    af::array img = af::constant(0, img_size_x, img_size_y, 4, 1, f32);
    return make_image(metainfo, img, save, frame_n, rotate);
}


af::array Painter::make_image(af::array metainfo, 
    af::array img, bool save, 
    int* frame_n, bool rotate) const
{
    int img_size_x = target_image.dims(0);
    int img_size_y =  target_image.dims(1);

    for (int n=0; n<img_size_x; n++)
    { 
        af::array mbrush = brush;

        float angle = af::sum<float>(metainfo(n, 2));
        af::array x = metainfo(n, 0);
        af::array y = metainfo(n, 1);

        af::array _target_img = target_image;

        img = ifs::add_imgs(mbrush, img, x, y, 1, angle,
            [&_target_img](af::array mbrush, af::array _1, af::array x, af::array y, af::array _2)
            {
                int size_x = mbrush.dims(0);
                int size_y = mbrush.dims(1);

                af::array mid_x = x(size_x/2);
                af::array mid_y = y(size_y/2);

                // apply color
                mbrush(af::span, af::span, af::seq(3), af::span) *= 
                    af::tile(_target_img(mid_x, mid_y, af::seq(3)), size_x, size_y);
                return mbrush;
            });

        if (save && (
            n == img_size_x - 1 || n % 50 == 0))
        {
            af::array mimg = (img * 255).as(u8);
            mimg = af::resize(0.5f, mimg);
            std::string filename = "../imgs/process/" + 
                std::to_string(*frame_n++) + ".png";
            af::saveImageNative(filename.c_str(), mimg);
        }
    }
    
    return img;
}


const af::array Painter::fitness_func(af::array coords)
{
    // coords is pop_size x dna_size_x x 4 x 1
    af::array x = coords(af::span, af::span, 0) * target_image.dims(0);
    af::array y = coords(af::span, af::span, 1) * target_image.dims(1);

    af::array grad = 2 * ifs::PI * coords(af::span, af::span, 2) - ifs::PI;

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
            af::approx2(1/(c_weights + grad_weights * img_gradient + 1),
                x(i, af::span), y(i, af::span)) *
            (af::approx2(img_gradient, x(i, af::span), y(i, af::span)) - 
            grad(i, af::span)));

        af::array variance_loss = // we don't want the same inputs 
            var_weights * .1f * af::tile(1/af::stdev(x(i, af::span), AF_VARIANCE_DEFAULT, 1) + 
            1/af::stdev(y(i, af::span), AF_VARIANCE_DEFAULT, 1), 1, coords.dims(1));

        results(i, af::span) = content_loss + variance_loss;        
    }

    return af::sum(
        -(af::pow(results, 2)
    ), 1); // sum every row
}


void Painter::run(bool save)
{
    float og_weights = var_weights;
    float mutation_rate = 0.001f;
    int frame_n = 0;

    for (int i=0; i<loops; i++)
    {
        GeneticAlgorithm gal(pop_size, dna_size_x, 
        dna_size_y, mutation_rate, iters);

        gal.run(*this);
        af::array best = gal.get_best();

        best = af::reorder(best, 1, 2, 0);

        // instead of just painting over the image
        // we should only paint parts with lower losses
        af::array img = make_image(best, current_img, 
            save, &frame_n, true);
        c_weights = calculate_weights(img);
        current_img = img;

        // adjust brush size for fine tunning
        if (i == loops / 2)
            brush = af::resize(0.25f, brush);
        if (i == 3 * loops / 4)
            brush = af::resize(0.8f, brush);

        
        if (i % 5 == 0 || i == iters - 1)
            std::cout << "finished iteration " << 
                i + 1 << std::endl;
    }

    var_weights = og_weights;
}


af::array Painter::calculate_weights(af::array c_img) const
{
    af::array _target_img = target_image;
    if (c_img.dims(2) != target_image.dims(2))
    {
        c_img = c_img(af::span, af::span, 
            af::seq(target_image.dims(2)));
    }

    af::array black = af::constant(0, c_img.dims());
    af::array mask = af::ceil(black + c_img);

    // set weights of unpainted areas as 1
    // weights of painted areas are the difference between
    // the current image and target
    af::array weights = (mask) * (c_img - target_image) + (1 - mask);
    return 3 * af::sum(af::pow(weights, 2), 2);
}


af::array Painter::get_target_img() const
{
    return target_image;
}

af::array Painter::get_current_img() const
{
    return current_img;
}

af::array Painter::get_current_weights() const
{
    return c_weights;
}