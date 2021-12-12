#pragma once

#include <functional>

#include "GridayTypes.hpp"
#include "GridayException.hpp"
#include "EnergyGrid.hpp"

class ChannelAnalyzer
    {
public:
    ChannelAnalyzer(EnergyGrid& energyGrid, GReal maxE, GReal dr);
    ~ChannelAnalyzer() = default;

    int getChannelDimension();
private:
    void analyze(GReal maxE, GReal dr);

private:
    EnergyGrid& mEnergyGrid;

    int mChannelDimension;
    };
