#include <iostream>
#include <arrayfire.h>
# include "include/genetic_algorithm.hpp"


af::array fitness_func(af::array A)
{
    return af::sum(af::sum(1.0f * (A), 1), 2); // sum every row
}


int main()
{
    int iters = 1000;
    int dna_size = 10;
    int pop_size = 200;
    float mutation_rate = 0.001f;
    float cross_amount = 0.8f;

    gfor(af::seq i, 5)
    {
        GeneticAlgorithm gal(pop_size, dna_size, dna_size,
        fitness_func, mutation_rate, cross_amount, iters);

        gal.run();
        af::array best = gal.get_best();
        float best_score = gal.get_best_score();
        af_print(best);
        std::cout << best_score << std::endl;
    }
        
    return 0;
}