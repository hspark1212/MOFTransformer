#pragma once

#include <random>
#include <chrono>

// Generate uniform random number in range [0,1)
class Random
    {
public:
    inline  Random();
    inline ~Random();

    inline double operator () ();
private:
    std::mt19937 mEngine;
    std::uniform_real_distribution<double> mDistribution;
    };

inline
Random::Random() :
    mEngine (std::chrono::system_clock::now().time_since_epoch().count()),
    mDistribution (0, 1)
    {

    }

inline
Random::~Random()
    {

    }

inline
double
Random::operator () ()
    {
    return mDistribution(mEngine);
    }
