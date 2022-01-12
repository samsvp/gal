#include <iostream>
#include <arrayfire.h>


class GeneticAlgorithm
{
public:
    GeneticAlgorithm(int pop_size, int dna_size_x,
        int dna_size_y, af::array (*fitness_func)(af::array), 
        float mutation_rate, float cross_amount,
        int iters);
    ~GeneticAlgorithm();
    void run();
    af::array get_best();
    float get_best_score();
    float cross_amount;
    float mutation_rate;

private:
    int iters;
    int dna_size_x;
    int dna_size_y;
    int pop_size;
    float best_score = 0;
    af::array best;
    // each row is a member
    // number of columns are genes
    af::array population; 
    af::array (*fitness_func)(af::array);

    void selection();
    void crossover();
    void renew(af::array row);
    void mutate();
    void fitness_score();
};


GeneticAlgorithm::GeneticAlgorithm(int _pop_size, 
        int dna_size_x, int dna_size_y, af::array (*fitness_func)(af::array), 
        float mutation_rate, float cross_amount, int iters) :
            dna_size_x(dna_size_x), dna_size_y(dna_size_y),
            mutation_rate(mutation_rate), fitness_func(fitness_func),
            cross_amount(cross_amount), iters(iters)
{
    pop_size = _pop_size % 2 == 0 ? _pop_size : _pop_size + 1;
    // creates a random population
    population = af::randu(pop_size, dna_size_x, dna_size_y);

    //af_print(population);
}


GeneticAlgorithm::~GeneticAlgorithm()
{

}


void GeneticAlgorithm::run()
{
    for (int i = 0; i < iters; i++)
    {
        selection();
        crossover(); 
        mutate();
    }
}


void GeneticAlgorithm::selection()
{
    af::array scores = fitness_func(population);
    
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
    renew(pop_best);

    if (pop_best_score > best_score)
    {
        best = pop_best;
        best_score = pop_best_score;
        std::cout << best_score << std::endl;
        //af_print(best);
    }
}


void GeneticAlgorithm::crossover()
{
    af::array p1 = population(af::seq(0, af::end, 2), af::span, af::span);
    af::array p2 = population(af::seq(1, af::end, 2), af::span, af::span);
    af::array p3 = p2;
    
    af::array r = af::randu(pop_size / 2, dna_size_x, dna_size_y);
    af::array idxs = r > cross_amount;

    p2 = idxs * p1 + (1 - idxs) * p2;
    p1 = idxs * p3 + (1 - idxs) * p1;

    population(af::seq(0, af::end, 2), af::span, af::span) = p1;
    population(af::seq(1, af::end, 2), af::span, af::span) = p2;
}


void GeneticAlgorithm::renew(af::array best)
{
    af::array r = af::randu(pop_size, dna_size_x, dna_size_y);
    af::array idxs_replace = r < 0.5f;

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
