#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

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
    const char* obj_dir = parse_option("-d", "../imgs/test/Selos", argc, argv);

    int m_int = parse_option("-i", 1, argc, argv);
    float m_float = parse_option("-f", 2.5, argc, argv);

    std::cout << m_int << std::endl;
    std::cout << m_float + 1 << std::endl;

    return 0;
}