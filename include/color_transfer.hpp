#pragma once

#include <cstdio>
#include <string>
#include <iostream>
#include <arrayfire.h>


namespace color_transfer
{
    namespace
    {
        const char* OUT_PATH = "tmp.jpg";
    }


    /*
    * Calls python color transfer functions and
    * returns the results
    */
    af::array color_transfer(
        std::string content_path, std::string style_path, std::string function)
    {
        // the script will create a tmp file
        std::string command = "python scripts/color_transfer.py " + 
            content_path + " " + style_path + " " + function;
        int status = system(
            (command).c_str());
        
        if (status != 0)
        {
            std::cout << "Python call failed" << std::endl;
            std::cout << "Error number: " << status << std::endl;
        }

        af::array img = af::loadImage(OUT_PATH, 1) / 255.f;
        
        // remove the tmp file created
        status = remove(OUT_PATH);
        if (status != 0)
        {
            std::cout << "Error deleting " << OUT_PATH << std::endl;
            std::cout << "Error number: " << status << std::endl;
        }

        return img;
    }

    af::array lab_transfer(std::string content_path, std::string style_path)
    {
        return color_transfer(content_path, style_path, "1");
    }

    af::array hist_transfer(std::string content_path, std::string style_path)
    {
        return color_transfer(content_path, style_path, "0");
    }
}