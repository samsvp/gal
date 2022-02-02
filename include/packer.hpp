#pragma once

#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <arrayfire.h>

#include "image_functions.hpp"
#include "genetic_algorithm.hpp"


/*
 * Recreates the given image using
 * the given objects
 */
class Packer : public Score
{
public:
    Packer(const char* target_path,
        std::vector<std::string> objs_path,
        float scale);
    ~Packer();

    const af::array fitness_func(af::array coords) override;

    /*
     * Runs the algorithm and returns the best
     * solution.
     */
    af::array run(int pop_size, int max_objs, 
        float mutation_rate, int iters=100, bool show_cost=false);

    // cost function weights
    float area_weight = 800;
    float cost_weight = 50;
    float fill_weight = 5;
    float angle_weight = 500;

private:
    /*
     * Metainfo is a Nx5 array containing:
     * (x,y,scale,obj_index,angle), all within the range of 0-1
     */
    af::array make_image(af::array coords) const;
    af::array make_image_bw(af::array coords) const;
    std::vector<af::array> object_set; // make a pure af::array later
    std::vector<af::array> objects_bw; // make a pure af::array later
    std::vector<af::array> objects; // make a pure af::array later
    af::array target_img;
};


Packer::Packer(const char* target_path,
    std::vector<std::string> objs_path,
    float scale)
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
        std::cout << "Loaded image " << obj_path << std::endl;
        af::array _img = af::loadImage(obj_path.c_str(), 1) / 255.f;
         // resize just to hold less stuff
        af::array _res_img = af::resize(scale, _img);
        object_set.push_back(_res_img);
    }
}


Packer::~Packer()
{

}


af::array Packer::run(int pop_size, int max_objs, 
    float mutation_rate, int iters, bool show_cost)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    // load random objects from the set into objects
    for (int i=0; i<max_objs; i++)
    {
        std::uniform_int_distribution<std::mt19937::result_type> 
            dist6(0, object_set.size() - 1);
        int r = dist6(rng);
        objects.push_back(object_set[r]);
        af::array bw_obj = (object_set[r] > 0.01);
        objects_bw.push_back(bw_obj(af::span, af::span, 0));
    }    

    GeneticAlgorithm gal(pop_size, max_objs, 4,
        mutation_rate, iters);

    gal.run(*this);
    af::array best = gal.get_best();

    best = af::reorder(best, 1, 2, 0);

    if (show_cost)
    {
        af::array img = make_image_bw(best);

        const af::array bw_image = (img(af::span, af::span, 0) > 0.001f);
        const af::array bw_target = (target_img > 0.001f);
        af::array cost = (bw_image + target_img) * (!bw_image + !target_img);

        af::Window wnd("Preliminary result");
            while (!wnd.close()) wnd.image(cost);
    }

    return make_image(best);
}


af::array Packer::make_image(af::array metainfo) const
{
    int img_size_x = target_img.dims(0);
    int img_size_y = target_img.dims(1);

    // kind of ineficient
    // it would be better if we could create
    // all instances of the population 
    // at the same time
    af::array img = af::constant(0, img_size_x, img_size_y, 4);

    for (int i=0; i<metainfo.dims(0); i++)
    {
        af::array foreground = af::resize(af::sum<float>(0.7f * metainfo(i, 2)+0.3), objects[i]);

        float angle = af::sum<float>(metainfo(i, 3));
        af::array x = metainfo(i, 0);
        af::array y = metainfo(i, 1);
        img = ifs::add_imgs(foreground, img, x, y, 1, angle);
    }
    
    return img;
}


af::array Packer::make_image_bw(af::array coord) const
{     
    int img_size_x = target_img.dims(0);
    int img_size_y = target_img.dims(1);
    
    af::array bw_img = af::constant(0, img_size_x, img_size_y, 1);
    af::array bw_target = (target_img > 0.01f);

    for (int i=0; i<coord.dims(0); i++)
    {
        af::array foreground = af::resize(
            af::sum<float>(0.7f * coord(i, 2)+0.3f), objects_bw[i]);;

        int size_x = foreground.dims(0);
        int size_y = foreground.dims(1);

        float angle = af::sum<float>(coord(i, 3));
        af::array _x = coord(i, 0);
        af::array _y = coord(i, 1);

        af::array x = af::seq(size_x) + 
            af::tile(_x * img_size_x, size_x);
        af::array y = af::seq(size_y) + 
            af::tile(_y * img_size_y, size_y);

        af::array n_img = bw_img;
        n_img(x, y, af::span, af::span) = 
            bw_img(x, y, af::span, af::span) + foreground;

        bw_img = n_img;
    }
    
    return bw_img;
}


// coords (pop_size, max_objs, 4, 1)
const af::array Packer::fitness_func(af::array coords)
{
    int pop_size = coords.dims(0);
    af::array costs = af::constant(0, pop_size, f32);
    
    for (int j = 0; j < pop_size; j++)
    {
        // make image
        // kind of ineficient
        // it would be better if we could create
        // all instances of the population 
        // at the same time
        af::array coord = af::reorder(coords(j, af::span), 1, 2, 0);
        
        af::array bw_img = make_image_bw(coord);
        af::array bw_target = (target_img > 0.01f);

        // punish for not filling the inside area 
        af::array area_cost = area_weight * af::sum(af::sum(bw_target * !bw_img));
        // punish for filling the outside area
        af::array cost = cost_weight * af::sum(af::sum(!bw_target * bw_img));

        costs(j) = cost + area_cost;
    }
    
    return -costs;
}
