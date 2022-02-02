#pragma once


#include <iostream>
#include <arrayfire.h>


class Score
{
public:
    const virtual af::array fitness_func(af::array)=0;
};


class GeneticAlgorithm
{
public:
    GeneticAlgorithm(int pop_size, int dna_size_x,
        int dna_size_y, float mutation_rate, int iters);
    ~GeneticAlgorithm();
    /*
     * Runs the algorithm
     */
    void run(Score& score);
    /*
     * Returns the best population individual
     */
    af::array get_best();
    /* 
     * Returns the best population score
     */
    float get_best_score();
    float mutation_rate;

private:
    int iters;
    int dna_size_x;
    int dna_size_y;
    int pop_size;
    float best_score = -100000000;
    /*
     * Best induvidual of all the generations
     */
    af::array best;
    // each row is a member
    // number of columns are genes
    af::array population;

    /*
     * Calculates fitness score and renews population
     * Function to compute the fitness score. It must
     * receive the populationa and return the score for each member.
     * The algorithm tries to maxime this function
     */
    void selection(Score& score);
    /*
     * Performs crossover on the population. Each
     * individual genes are combined with another
     */
    void crossover();
    /*
     * "Breeds" the best specimen(the one with the highst score)
     * with the rest of the population
     */
    void crossover(af::array row);
    /*
     * Mutates the population
     */
    void mutate();
    /*
     * Calculates the fitness score
     */
    void fitness_score();
};


GeneticAlgorithm::GeneticAlgorithm(int _pop_size, 
        int dna_size_x, int dna_size_y, float mutation_rate, 
        int iters) :
            dna_size_x(dna_size_x), dna_size_y(dna_size_y),
            mutation_rate(mutation_rate), iters(iters)
{
    pop_size = _pop_size % 2 == 0 ? _pop_size : _pop_size + 1;
    // creates a random population
    population = af::randu(pop_size, dna_size_x, dna_size_y);
}


GeneticAlgorithm::~GeneticAlgorithm()
{

}


void GeneticAlgorithm::run(Score& score)
{
    for (int i = 0; i < iters; i++)
    {
        selection(score);
        mutate();

        #ifndef NDEBUG
        if (i % 20 == 0)
            std::cout << "iteration: " << i << std::endl;
        #endif
    }
}


void GeneticAlgorithm::selection(Score& score)
{
    af::array scores = score.fitness_func(population);
    
    af::array pop_best_scores;
    af::array pop_best_idx;
    af::max(pop_best_scores, pop_best_idx, scores, 0);

    float pop_best_score = pop_best_scores.scalar<float>();

    // normalize scores
    scores -= af::min(scores).scalar<float>();
    scores /= af::sum(scores).scalar<float>();

    // this is stupid, arrayfire doesn't let
    // me get the first element
    int idx = af::sum<int>(pop_best_idx(0));
    
    af::array pop_best = population(idx,
        af::span, af::span);

    crossover(pop_best);

    if (pop_best_score > best_score)
    {
        best = pop_best;
        best_score = pop_best_score;

        #ifndef NDEBUG
        std::cout << "New best score " << best_score << std::endl;
        #endif
    }
}


void GeneticAlgorithm::crossover(af::array best)
{
    af::array r = af::randu(pop_size, dna_size_x, dna_size_y);
    af::array idxs_replace = r < 0.5f; // should be based on score

    population = 
        idxs_replace * af::tile(best, pop_size) + 
        (1 - idxs_replace) * population;
}


void GeneticAlgorithm::mutate()
{
    af::array r = af::randu(pop_size, dna_size_x, dna_size_y);
    af::array u = af::randu(pop_size, dna_size_x, dna_size_y);

    // change to a random value if r < than
    // the mutation rate, else continue with
    // the same value
    population = (r < mutation_rate) * u + 
        (r > mutation_rate) * population;
}


af::array GeneticAlgorithm::get_best()
{
    return best;
}


float GeneticAlgorithm::get_best_score()
{
    return best_score;
}
