#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <arrayfire.h>

#include "include/genetic_algorithm.hpp"
#include "include/packer.hpp"


namespace fs = std::filesystem;

int main()
{
    std::vector<std::string> obj_pths;
    std::string path = "../imgs/Selos";

    for (const auto & entry : fs::directory_iterator(path))
        obj_pths.push_back(entry.path());

    int pop_size = 100;
    int max_objs = 20;
    int iters = 100;
    float mutation_rate = 0.001f;
    Packer packer("../imgs/reserva.png", obj_pths);
    af::array current_img = packer.run(pop_size, max_objs, mutation_rate, iters);

    af::Window wnd(800, 800, "Preliminary result");
        while (!wnd.close()) wnd.image(current_img);

    return 0;
}