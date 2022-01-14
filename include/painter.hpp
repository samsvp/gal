#pragma once

#include <iostream>
#include <arrayfire.h>
#include "genetic_algorithm.hpp"


namespace Painter
{

    float PI = 3.141592653f;

    int pop_size = 100;
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

        brush = af::loadImage("../brushes/3.png", true) / 255.f;
        brush(af::span, af::span, af::seq(3)) += 0.f;
        brush = af::medfilt2(brush, 5, 5);
        
        brush = af::resize(0.02f, brush);
        std::cout << "brush dims " << brush.dims() << std::endl;
        std::cout << "target image " << target_image.dims() << std::endl;
    }


    af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask)
    {
        af::array tiled_mask;
        if (mask.dims(2) != foreground.dims(2))
            tiled_mask = tile(mask, 1, 1, foreground.dims(2));
        return foreground * tiled_mask + (1.0f-tiled_mask) * background;
    }


    /*
     * Metainfo is a Nx4 array containing:
     * (x,y,color,angle) all within the range of 0-1
     */
    af::array make_image(af::array metainfo)
    {
        int img_size_x = target_image.dims(0);
        int img_size_y =  target_image.dims(1);
        af::array img = af::constant(0, img_size_x, img_size_y, 4, 1, f32);

        std::cout << "metainfo dims: " << metainfo.dims() << std::endl;

        for (int n=0; n<metainfo.dims(0); n++)
        { 
            af::array mbrush = brush;// af::rotate(brush,
            //     // af::sum<float>(metainfo(n,3)) / 5, 1, 
            //     af::sum<float>(metainfo(n,3)) / 5, 0, 
            //     AF_INTERP_BICUBIC);

            int size_x = mbrush.dims(0);
            int size_y = mbrush.dims(1);
            // apply color
            mbrush(af::span, af::span, af::seq(3), af::span) = 
                af::tile(metainfo(n,2), size_x, size_y, 3);

            af::array x = af::seq(size_x) + 
                af::tile(metainfo(n, 0) * img_size_x, size_x);
            af::array y = af::seq(size_y) + 
                af::tile(metainfo(n, 1) * img_size_y, size_y);

            af::array mask = mbrush(af::span, af::span, -1) != 0;
            // af::array mask = mbrush(af::span, af::span, -1);

            img(x, y, af::span, af::span) = 
                alpha_blend(mbrush, img(x, y, af::span, af::span), mask);
        }
        
        return img;
    }


    af::array fitness_func(af::array coords)
    {
        // coords is pop_size x dna_size_x x 4 x 1
        af::array x = coords(af::span, af::span, 0) * target_image.dims(0);
        af::array y = coords(af::span, af::span, 1) * target_image.dims(1);
        af::array color = coords(af::span, af::span, 2);

        af::array results = af::constant(0, coords.dims(0), coords.dims(1));

        // we could add the previous inputs into the cost function
        // adding into the stdev part, so the colors will gradually
        // have more weight
        for(int i=0; i<pop_size; i++)
        {
            results(i, af::span) = af::abs(af::approx2(
                target_image, x(i, af::span), y(i, af::span)) - color(i, af::span)) +
                // we don't want the same inputs 
                .1f * af::tile(1/af::stdev(x(i, af::span), AF_VARIANCE_DEFAULT, 1) + 
                    1/af::stdev(y(i, af::span), AF_VARIANCE_DEFAULT, 1), 1, coords.dims(1));
        }

        return af::sum(
                -(af::pow(results, 2)
                // 1/(af::pow(target_image - paint_img, 2)+0.1f
            ), 1); // sum every row
    }


    void split_target_image()
    {
        af::array n_target = target_image;

        float mutation_rate = 0.001f;
        float cross_amount = 0.5f;
        int iters = 500;
        int dna_size_x = 512;
        int dna_size_y = 3;

        const int max_x = 5;
        const int max_y = 5;

        int cols = target_image.dims(0);
        int rows = target_image.dims(1);

        int step_x = cols / max_x;
        int step_y = rows / max_y;

        // dna_size_x, dna_size_y
        af::array dna = af::constant(0, dna_size_x * max_x * max_y, 3);
        af::array img = af::constant(0, cols, rows, 4, 1, f32);

        int i = 0;
        for (int x=0; x < max_x; x++)
        {
            for (int y = 0; y < max_y; y++)
            {
                af::array seq_x = af::seq(x*step_x, (x+1)*step_x);
                af::array seq_y = af::seq(y*step_y, (y+1)*step_y);
                af::array sub_img = n_target(seq_x, seq_y);
                
                target_image = sub_img;

                GeneticAlgorithm gal(pop_size, dna_size_x, dna_size_y,
                    fitness_func, mutation_rate, cross_amount, iters);

                gal.run();
                af::array best = gal.get_best();

                //results(af::span, af::seq(15), af::span) = gal.get_best();
                best = af::reorder(best, 1, 2, 0);
                
                // change to real coordinates
                best(af::span, 0) = best(af::span, 0) / max_x + 1./max_x * x;
                best(af::span, 1) = best(af::span, 1) / max_y + 1./max_y * y;

                dna(af::seq(i*dna_size_x,(i+1)*dna_size_x-1), af::span) = best;

                af::array results = make_image(best);
                i++;
                // af::Window wnd(800, 800, "Preliminary result");
                //     while (!wnd.close()) wnd.image(results);
            }
        }
        
        target_image = n_target;
        img = make_image(dna);

        af::Window wnd(400, 400, "Preliminary result");
            while (!wnd.close()) wnd.image(img);
        
        af::Window wnd2(400, 400, "OG");
            while (!wnd2.close()) wnd2.image(n_target);
    }


    void run()
    {
        // float f[] = {0.5, 0.45, 0.5, 0.5, 0.5, 0.5, 1, 0.};
        // af::array a(2, 4, f);

        // af_print(a);
        // af::array img = make_image(a);

        // af::Window wnd(800, 800, "Window");
        //     while (!wnd.close()) wnd.image(img);

        // af::array _a = af::randu(2, 4, 1, 20);
        // std::cout << "a dims: " << a.dims() << std::endl;
        // std::cout << "_a dims: " << _a.dims() << std::endl;
        // af::array _img = make_image(_a);
        
        float mutation_rate = 0.001f;
        float cross_amount = 0.5f;
        int iters = 10;

        results = af::randu(1, target_image.dims(0), target_image.dims(1));

        int dna_size_x = 2048;
        int dna_size_y = 3;

        af::array n_target = target_image;
        n_target = af::reorder(n_target, 2, 0, 1);

        GeneticAlgorithm gal(pop_size, dna_size_x, dna_size_y,
            fitness_func, mutation_rate, cross_amount, iters);

        gal.run();
        af::array best = gal.get_best();

        //results(af::span, af::seq(15), af::span) = gal.get_best();
        //std::cout << "wtf" << std::endl;
        best = af::reorder(best, 1, 2, 0);
        results = make_image(best);
        std::cout << results.dims() << std::endl;
        
        af::Window wnd(800, 800, "Window");
            while (!wnd.close()) wnd.image(results);
    }

};
