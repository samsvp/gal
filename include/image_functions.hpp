#pragma once

#include <functional>
#include <arrayfire.h>


namespace ifs
{
    const float PI = 3.14159;

    /*
     * Adds two images using the alpha channel
     */
    const af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask)
    {
        af::array tiled_mask;
        if (mask.dims(2) != foreground.dims(2))
            tiled_mask = tile(mask, 1, 1, foreground.dims(2));
        return foreground * tiled_mask + (1.0f-tiled_mask) * background;
    }


    af::array add_imgs(af::array& foreground, 
        af::array& background, af::array _x, 
        af::array _y, float scale,
        bool resize, bool rotate, float _angle,
        std::function<af::array(
            af::array&, af::array&, af::array&, af::array&, af::array&)> f,
        bool skip_f=0)
    {
        
        if (rotate)
        {
            float angle = 2 * PI * _angle - PI;
            foreground = af::rotate(foreground, angle, 0, 
                AF_INTERP_BICUBIC_SPLINE);
        }

        if (resize)
        {
            af::resize(scale, foreground, AF_INTERP_BILINEAR);
        }

        int img_size_x = background.dims(0);
        int img_size_y = background.dims(1);

        int size_x = foreground.dims(0);
        int size_y = foreground.dims(1);

        af::array x = af::seq(size_x) + 
            af::tile(_x * img_size_x, size_x);
        af::array y = af::seq(size_y) + 
            af::tile(_y * img_size_y, size_y);

        af::array mask = foreground(af::span, af::span, -1);

        if (!skip_f)
            foreground = f(foreground, background, x, y, mask);

        background(x, y, af::span, af::span) =
            alpha_blend(foreground, background(x, y, af::span, af::span), mask);

        return background;
    }


    af::array add_imgs(af::array& foreground, 
        af::array& background, af::array _x, 
        af::array _y, float scale,
        bool resize, bool rotate, float angle)
    {
        return add_imgs(foreground, background, _x, _y, scale,
            resize, rotate, angle, [](af::array& _1, af::array& _2,
            af::array& _3, af::array& _4, af::array& _5){return _1;}, 1);
    }
}