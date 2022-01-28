#pragma once

#include <vector>
#include <string>
#include <arrayfire.h>

#include "genetic_algorithm.hpp"


class Packer : public Score
{
public:
    Packer(const char* target_path,
        std::vector<std::string> objs_path);
    ~Packer();

    const af::array fitness_func(af::array coords) override;

    af::array run(int pop_size, int max_objs, 
        float mutation_rate, int iters=100);


private:
    const float PI = 3.14159;
    /*
     * Adds two images using the alpha channel
     */
    af::array alpha_blend(const af::array &foreground, 
        const af::array &background, const af::array &mask) const;
    const af::array make_image(af::array coords);
    std::vector<af::array> objects;
    af::array target_img;
};


Packer::Packer(const char* target_path,
    std::vector<std::string> objs_path)
{
    af::array _target_img = af::loadImage(target_path, 1) / 255.f;

    // we do this so that the rgb img takes alpha into account
    if (_target_img.dims(2) == 4)
        _target_img = _target_img(af::span, af::span, af::seq(3)) *
            af::tile(_target_img(af::span, af::span, 3), 1, 1, 3);

    target_img = af::rgb2gray(_target_img);
    // normalize values
    target_img /= af::max<float>(target_img);

    // load img objects
    for (std::string& obj_path : objs_path)
    {
        af::array _img = af::loadImage(obj_path.c_str(), 1) / 255.f;
         // resize just to hold less stuff
        af::array _res_img = af::resize(0.5f, _img);
        objects.push_back(_res_img);
    }
}


Packer::~Packer()
{

}


af::array Packer::run(int pop_size, int max_objs, 
    float mutation_rate, int iters)
{
    GeneticAlgorithm gal(pop_size, max_objs, 4,
        mutation_rate, iters);

    gal.run(*this);
    af::array best = gal.get_best();

    best = af::reorder(best, 1, 2, 0);
    
    af::array img = make_image(best);
    const af::array bw_image = (img(af::span, af::span, 3) > 0.01f);
    const af::array bw_target = (target_img > 0.01f);
    af::array cost = (bw_image + target_img) * (!bw_image + !target_img);

    af::Window wnd(800, 800, "Preliminary result");
        while (!wnd.close()) wnd.image(cost);

    return make_image(best);
}


const af::array Packer::make_image(af::array metainfo)
{
    int img_size_x = target_img.dims(0);
    int img_size_y = target_img.dims(1);


    // kind of ineficient
    // it would be better if we could create
    // all instances of the pop 
    // at the same time
    af::array img = af::constant(0, img_size_x, img_size_y, 4);

    af::array indexes = metainfo(af::span, 3) * objects.size();
    indexes = indexes.as(s32);

    for (int i=0; i<metainfo.dims(0); i++)
    {
        int idx = indexes(i).scalar<int>();

        af::array stamp = objects[idx];

        int size_x = stamp.dims(0);
        int size_y = stamp.dims(1);

        af::array x = af::seq(size_x) + 
            af::tile(metainfo(i, 0) * img_size_x, size_x);
        af::array y = af::seq(size_y) + 
            af::tile(metainfo(i, 1) * img_size_y, size_y);

        af::array mid_x = x(size_x/2);
        af::array mid_y = y(size_x/2);
        
        af::array mask = stamp(af::span, af::span, -1);

        img(x, y, af::span, af::span) = 
            alpha_blend(stamp, img(x, y, af::span, af::span), mask);
    }
    
    return img;
}


af::array Packer::alpha_blend(const af::array &foreground, 
    const af::array &background, const af::array &mask) const
{
    af::array tiled_mask;
    if (mask.dims(2) != foreground.dims(2))
        tiled_mask = tile(mask, 1, 1, foreground.dims(2));
    return foreground * tiled_mask + (1.0f-tiled_mask) * background;
}


const af::array Packer::fitness_func(af::array coords)
{
    int pop_size = coords.dims(0);
    af::array costs = af::constant(0, pop_size, f32);
    for (int i=0; i<pop_size; i++)
    {
        af::array coord = af::reorder(coords(i, af::span), 1, 2, 0);
        af::array img = make_image(coord);

        // turn every transparent pixel into black
        // and every non transparent pixel into white
        const af::array bw_image = (img(af::span, af::span, 3) > 0.01f);
        const af::array bw_target = (target_img > 0.01f);
        // we punish the image if it puts objects outsite
        // but reward it for putting objects inside the picture
        af::array cost = (bw_image + target_img) * (!bw_image + !target_img);

        // sum over both dimensions
        costs(i) = af::sum(af::sum(cost));
    }
    
    return -costs;
}