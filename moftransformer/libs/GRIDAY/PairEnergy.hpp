#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <functional>

#include "GridayTypes.hpp"
#include "Vector.hpp"
#include "Cell.hpp"

class PairEnergy
    {
public:
    using PairEnergyPtr = std::unique_ptr<PairEnergy>;

    PairEnergy() = default;
    virtual ~PairEnergy() = default;

    virtual void setSimulationBox(const Cell& box) = 0;
    virtual GReal calculate(const Vectors& r1, const Vectors& r2) = 0;

    virtual std::unique_ptr<PairEnergy> clone() = 0;
    virtual void print() = 0;
    virtual std::string getName() = 0;
    virtual GReal getRcut() = 0;
    };

template <typename E>
PairEnergy::PairEnergyPtr mixPairEnergy(PairEnergy& e1, PairEnergy& e2);
