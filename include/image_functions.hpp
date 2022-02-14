#pragma once

#include <vector>
#include <functional>
#include <arrayfire.h>


namespace ifs
{
    const float PI = 3.14159;

    // ---------------------------------------------
    // Definitions
    // ---------------------------------------------

    /*
     * Adds two images using the alpha channel
     */
    const af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask);

    /*
     * Does a batch translation on the given image
     * 't' must be of size 2 x 1 x 1 x b
     * where imgs is of size w x h x c x b
     * where 'b' is the batch size
     */
    af::array translate(af::array imgs, af::array t);

    /*
     * Transforms a vector of images into an array
     * of same sized images
     */
    af::array vec_to_array(std::vector<af::array> v, int channels);

    /*
     * Puts one image above another. A function
     * can be passed to perform a transformation
     * on the calculated parameters before merging
     */
    af::array add_imgs(af::array& foreground, 
        af::array& background, af::array _x, 
        af::array _y, float scale,
        bool resize, bool rotate, float angle);

    af::array add_imgs(af::array& foreground, 
        af::array& background, af::array _x, 
        af::array _y, float scale,
        bool resize, bool rotate, float _angle,
        std::function<af::array(
            af::array&, af::array&, af::array&, af::array&, af::array&)> f,
        bool skip_f);



    // ---------------------------------------------
    // Implementations
    // ---------------------------------------------

    const af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask)
    {
        af::array tiled_mask;
        if (mask.dims(2) != foreground.dims(2))
            tiled_mask = tile(mask, 1, 1, foreground.dims(2));
        return foreground * tiled_mask + (1.0f-tiled_mask) * background;
    }


    af::array translate(af::array imgs, af::array t)
    {
        // translation matrix
        int b = imgs.dims(3);
        af::array T = af::identity(3, 3, 1, b);
        // yx axis translations
        T(0, 2, af::span) = t(0, af::span);
        T(1, 2, af::span) = t(1, af::span);

        af::array translated = af::transform(imgs, T.T(), 
            imgs.dims(0), imgs.dims(1));
        return translated;
    }


    af::array vec_to_array(std::vector<af::array> imgs_vec, int channels=4)
    {

        // computes maximum width and height
        auto cmp_size = [](af::array a1, af::array a2, int d){ return a1.dims(d) < a2.dims(d); };

        int max_h = (*std::max_element(imgs_vec.begin(), imgs_vec.end(), 
            [&cmp_size](auto a1, auto a2) { return cmp_size(a1, a2, 0); })).dims(0);
        int max_w = (*std::max_element(imgs_vec.begin(), imgs_vec.end(),
            [&cmp_size](auto a1, auto a2) { return cmp_size(a1, a2, 1); })).dims(1);
        
        // creates array to hold all images
        af::array arr = af::constant(0, max_h, max_w, channels, imgs_vec.size());

        // returns the begin and end padding values needed to have the image be of max shape
        auto get_pad_values = [](af::array img, int max, int d) {
            int delta = max - img.dims(d);
            int pad_begin = delta / 2;
            int pad_end = delta % 2 ?  delta / 2 + 1 : delta / 2;
            return std::make_pair(pad_begin, pad_end);
        };

        
        for (int i=0; i<imgs_vec.size(); i++)
        {
            // get padding values
            af::array img = imgs_vec[i];
            std::pair<int, int> d0 = get_pad_values(img, max_h, 0);
            std::pair<int, int> d1 = get_pad_values(img, max_w, 1);
            
            af::dim4 begin_padding = af::dim4(d0.first, d1.first, 0, 0);
            af::dim4 end_padding = af::dim4(d0.second, d1.second, 0, 0);
            
            // pad images
            af::array p_img = af::pad(img, begin_padding, end_padding, AF_PAD_ZERO);
            arr(af::span, af::span, af::span, i) = p_img;
        }

        return arr;
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