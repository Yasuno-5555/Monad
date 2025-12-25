#ifndef MONAD_GRID_GENERATOR_H
#define MONAD_GRID_GENERATOR_H

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

class GridGenerator {
public:
    static std::vector<double> generate(const char* kind, double min_val, double max_val, int size, double curvature = 1.0);

private:
    static std::vector<double> uniform(double min_val, double max_val, int size);
    static std::vector<double> log_spaced(double min_val, double max_val, int size, double curvature);
};

#endif // MONAD_GRID_GENERATOR_H
