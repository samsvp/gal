#include <cstring>
#include <vector>
#include <string>
#include <sstream>
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


template<typename T>
T parse_option(const char* option, T deflt, 
    int argc, char**argv)
{
    for (int i=1; i<argc; i++)
    {
        if (std::strcmp(argv[i], option) == 0)
        {
            std::stringstream ss(argv[i+1]);
            T t;
            ss >> t;
            return t;
        }
    }
    return deflt;
}


int main(int argc, char **argv)
{

    /* parse arguments */
    // files
    const char* obj_dir = parse_option("-d", "../imgs/test/Selos", argc, argv);
    const char* img_path = parse_option("-t", "../imgs/reserva_t.png", argc, argv);
    const char* save_name = parse_option("-s", "../imgs/packer_out.png", argc, argv);
    int callback = parse_option("-c", 1, argc, argv);

    // metaparameters
    float scale = parse_option("-r", 0.05f, argc, argv);
    int pop_size = parse_option("-p", 100, argc, argv);
    int max_objs = parse_option("-o", 120, argc, argv);
    int iters = parse_option("-i", 800, argc, argv);
    float mutation_rate = parse_option("-m", 0.001f, argc, argv);
    bool angle = parse_option("-a", 1, argc, argv);

    // weights
    float area_weight = parse_option("-a", 800, argc, argv);
    float out_weight = parse_option("-w", 50, argc, argv);

    std::cout << "\nStarting with parameters: scale " << scale <<
        ", population size " << pop_size << ", max objects " << max_objs <<
        ", iterations " << iters << ", mutation rate " << mutation_rate << 
        ", callback: " << callback << std::endl;

    std::cout << "\nWeights :area weight " <<
        area_weight << ", out weight " << out_weight << "\n" << std::endl;

    std::cout << "\nObjects directory " << obj_dir << ", target image path " <<
        img_path << ", output name " << save_name << "\n" << std::endl;

    std::vector<std::string> obj_pths;
    for (const auto & entry : fs::directory_iterator(obj_dir))
        obj_pths.push_back(entry.path());

    Packer packer(img_path, obj_pths, scale, angle);
    packer.area_weight = area_weight;
    packer.out_weight = out_weight;
    
    af::array current_img = packer.run(pop_size, max_objs, mutation_rate, iters, 0, callback);
    
    packer.save(save_name);

    af::Window wnd("Preliminary result");
        while (!wnd.close()) wnd.image(current_img);

    return 0;
}