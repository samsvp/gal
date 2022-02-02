#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <arrayfire.h>

#include "include/genetic_algorithm.hpp"
#include "include/packer.hpp"


namespace fs = std::filesystem;

const char* parse_option(const char* option, const char* deflt, 
    int argc, char**argv)
{
    for (int i=1; i<argc; i++)
    {
        if (std::strcmp(argv[i], option) == 0)
            return argv[i+1];
    }
    return deflt;
}

int main(int argc, char **argv)
{

    // parse arguments
    const char* obj_dir = parse_option("-o", "../imgs/test/Selos", argc, argv);
    const char* img_path = parse_option("-i", "../imgs/reserva_t.png", argc, argv);
    const char* save_name = parse_option("-s", "../imgs/packer_out.png", argc, argv);
    

    std::vector<std::string> obj_pths;
    for (const auto & entry : fs::directory_iterator(obj_dir))
        obj_pths.push_back(entry.path());

    float scale = 0.05f;
    int pop_size = 100;
    int max_objs = 120;
    int iters = 800;
    float mutation_rate = 0.001f;
    Packer packer(img_path, obj_pths, scale);
    af::array current_img = packer.run(pop_size, max_objs, mutation_rate, iters);

    af::array mimg = (current_img * 255).as(u8);
    af::saveImageNative("../imgs/test1.png", mimg);

    af::Window wnd("Preliminary result");
        while (!wnd.close()) wnd.image(current_img);

    return 0;
}