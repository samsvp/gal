#pragma once

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
        bool norm_imgs);
    ~Packer();

    const af::array fitness_func(af::array coords) override;

    /*
     * Runs the algorithm and returns the best
     * solution.
     */
    af::array run(int pop_size, int max_objs, 
        float mutation_rate, int iters=100);

    // cost function weights
    float fill_weight = 5;
    float grad_weight = 0.1;
    float angle_weight = 500;
    float obj_var_weight = 1000;
    float scale_var_weight = 1000;


private:
    /*
     * Metainfo is a Nx5 array containing:
     * (x,y,scale,obj_index,angle), all within the range of 0-1
     */
    af::array make_image(af::array coords) const;
    std::vector<af::array> objects; // make a pure af::array later
    af::array target_img;
    af::array gradient_img;
};


Packer::Packer(const char* target_path,
    std::vector<std::string> objs_path,
    bool norm_imgs)
{
    af::array _target_img = af::loadImage(target_path, 1) / 255.f;

    // we do this so that the rgb img takes alpha into account
    if (_target_img.dims(2) == 4)
        _target_img = _target_img(af::span, af::span, af::seq(3)) *
            af::tile(_target_img(af::span, af::span, 3), 1, 1, 3);

    target_img = af::rgb2gray(_target_img);
    // normalize values
    target_img /= af::max<float>(target_img);

    // image edges gradient
    af::array dx;
    af::array dy;
    // work on blurred img to have a smoother gradient
    af::sobel(dx, dy, af::medfilt2(target_img, 5, 5));
    gradient_img = af::abs(af::atan2(dy, dx));


    // load img objects
    for (std::string& obj_path : objs_path)
    {
        std::cout << obj_path << std::endl;
        af::array _img = af::loadImage(obj_path.c_str(), 1) / 255.f;
         // resize just to hold less stuff
        af::array _res_img = af::resize(0.1f, _img);
        objects.push_back(_res_img);
    }

    if (!norm_imgs) return;

    // rescale images to have around the same size
    // get x dims
    std::vector<int> x_dims;
    std::transform(objects.begin(), objects.end(), 
        std::back_inserter(x_dims), [](af::array img) {return img.dims(0);});

    // find mean x dims
    float mean = std::reduce(x_dims.begin(), x_dims.end())/(float)x_dims.size();

    // rescale
    std::transform(objects.begin(), objects.end(), 
        objects.begin(), 
        [&mean](af::array img) {return af::resize(mean / img.dims(0), img);});


    std::cout << "Images resized to:" << std::endl;
    for (auto &img: objects)
        std::cout << img.dims() << std::endl;
}


Packer::~Packer()
{

}


af::array Packer::run(int pop_size, int max_objs, 
    float mutation_rate, int iters)
{
    GeneticAlgorithm gal(pop_size, max_objs, 5,
        mutation_rate, iters);

    gal.run(*this);
    af::array best = gal.get_best();

    best = af::reorder(best, 1, 2, 0);
    
    af::array img = make_image(best);
    const af::array bw_image = (img(af::span, af::span, 3) > 0.01f);
    const af::array bw_target = (target_img > 0.01f);
    af::array cost = (bw_image + target_img) * (!bw_image + !target_img);

    af::Window wnd("Preliminary result");
        while (!wnd.close()) wnd.image(cost);

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

    af::array indexes = metainfo(af::span, 3) * objects.size();
    indexes = indexes.as(s32);

    for (int i=0; i<metainfo.dims(0); i++)
    {
        int idx = indexes(i).scalar<int>();

        af::array stamp = af::resize(af::sum<float>(0.7f * metainfo(i, 2)+0.3), objects[idx]);

        float angle = af::sum<float>(metainfo(i, 4));
        af::array x = metainfo(i, 0);
        af::array y = metainfo(i, 1);
        img = ifs::add_imgs(stamp, img, x, y, 1, angle);
    }
    
    return img;
}


const af::array Packer::fitness_func(af::array coords)
{
    int pop_size = coords.dims(0);
    af::array costs = af::constant(0, pop_size, f32);
    for (int i=0; i<pop_size; i++)
    {
        af::array coord = af::reorder(coords(i, af::span), 1, 2, 0);
        
        // image should actually be made inside here
        // that's because we must punish overlapping objs
        af::array img = make_image(coord);

        // turn every transparent pixel into black
        // and every non transparent pixel into white
        const af::array bw_image = (img(af::span, af::span, 3) > 0.01f);
        const af::array bw_target = (target_img > 0.01f);
        const af::array bw_gradient = (gradient_img > 0.01f);
        // we punish the image if it puts objects outsite
        // but reward it for putting objects inside the picture
        af::array cost = (fill_weight * bw_image + target_img) * (!bw_image + !target_img);
        af::array grad_cost = grad_weight * bw_image * !bw_gradient; // punish for not filling the edges
        
        af::array x = coord(af::span, 0);
        af::array y = coord(af::span, 1);
        
        /*
         * make the constants hyperparameters
         */
        // calculate angle difference
        af::array angles = 2 * ifs::PI * (coord(af::span, 4)) - ifs::PI;
        af::array angle_cost = angle_weight * af::sum(af::pow(angles - af::approx2(gradient_img, x, y), 2));

        // it is also undesirable to use the same img every time
        af::array ivariance_loss = 
            obj_var_weight * 1/(af::stdev(coord(af::span, 3), AF_VARIANCE_DEFAULT));

        // don't choose the same scales
        af::array svariance_loss = 
            scale_var_weight * 1/(af::stdev(coord(af::span, 2), AF_VARIANCE_DEFAULT));

        // total cost
        costs(i) = af::sum(af::sum(cost + grad_cost)) + angle_cost + ivariance_loss + svariance_loss;
    }
    
    // we need to frame it as a maximization problem
    return -costs;
}