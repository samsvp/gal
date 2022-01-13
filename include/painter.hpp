#pragma once

#include <iostream>
#include <arrayfire.h>
#include "genetic_algorithm.hpp"


namespace Painter
{

    float PI = 3.141592653f;

    int pop_size = 500;
    af::array target_split;

    af::array target_image;
    af::array brush;
    af::array results;

    void painter(const char *img_path, const char *brush_path)
    {
        // downsample
        target_image = af::loadImage(img_path) / 255.f;
        target_image = af::resize(0.5f, target_image);
        target_image = af::medfilt2(target_image, 5, 5);
        target_image = af::resize(0.5f, target_image);
        target_image = af::medfilt2(target_image, 5, 5);

        brush = af::loadImage("../brushes/1.png", true) / 255.f;
        brush(af::span, af::span, af::seq(3)) += 0.f;
        brush = af::medfilt2(brush, 5, 5);
        
        brush = af::resize(0.5f, brush);
        std::cout << "brush dims " << brush.dims() << std::endl;
        std::cout << "target image " << target_image.dims() << std::endl;
    }


    af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask)
    {
        af::array tiled_mask;
        if (mask.dims(2)!=foreground.dims(2))
            tiled_mask = tile(mask,1,1,foreground.dims(2));
        return foreground*tiled_mask+(1.0f-tiled_mask)*background;
    }

    /*
     * Metainfo is a Nx4 array containing:
     * (x,y,color,angle) all within the range of 0-1
     */
    af::array make_image(af::array metainfo)
    {
        ///
        /// Needs to implement color
        ///
        int img_size = 500;
        af::array img = af::constant(0, img_size, img_size, 4, f32);

        for (int n=0; n<metainfo.dims(0); n++)
        {
            int x = metainfo(n, 0).scalar<float>() * img_size;
            int y = metainfo(n, 1).scalar<float>() * img_size;

            af::array mbrush = af::rotate(brush, 
                metainfo(n,3).scalar<float>() * 2 * PI, 0, 
                AF_INTERP_BICUBIC);

            int size_x = mbrush.dims(0);
            int size_y = mbrush.dims(1);
            af::array mask = mbrush(af::span, af::span, -1) != 0;

            af::seq slice_x = af::seq(x, x + size_x - 1);
            af::seq slice_y = af::seq(y, y + size_y - 1);
            img(slice_x, slice_y, af::span) = 
                alpha_blend(mbrush, img(slice_x, slice_y, af::span), mask);
        }
        
        return img;
    }


    // af::array fitness_func(af::array paint_img)
    // {
    //     return af::sum(
    //         af::sum(
    //             1/(af::pow(target - paint_img, 2)+0.1f
    //         ), 1), 2); // sum every row
    // }


    void run()
    {
        float f[] = {0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0., 0.2};
        af::array a(2, 4, f);
        af_print(a);
        af::array img = make_image(a);

        af::Window wnd(800, 800, "Window");
             while (!wnd.close()) wnd.image(img);

        // float mutation_rate = 0.001f;
        // float cross_amount = 0.5f;
        // int iters = 500;

        // results = af::randu(1, target_image.dims(0), target_image.dims(1));

        // af::array n_target = target_image(af::seq(15), af::span, af::span);
        // n_target = af::reorder(n_target, 2, 0, 1);

        // int dna_size_x = 15;
        // int dna_size_y = target_image.dims(1);

        // GeneticAlgorithm gal(pop_size, dna_size_x, dna_size_y,
        //     n_target, fitness_func, mutation_rate, cross_amount, iters);

        // gal.run();
        // auto best =  gal.get_best();

        // results(af::span, af::seq(15), af::span) = gal.get_best();
        // af_print(n_target - gal.get_best())
        // //results = af::reorder(results, 1, 2, 0);
        
        
        // results = af::reorder(results, 1, 2, 0);
        // std::cout << "results dims: " << results.dims() << std::endl;
        // std::cout << "best dims: " << best.dims() << std::endl;

        // af::Window wnd(800, 800, "Window");
        //     while (!wnd.close()) wnd.image(results);
    }

};
