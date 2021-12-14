#pragma once

// For now, only consider truncated Lennard-Jones potential...

#include <functional>
#include <vector>
#include <memory>
#include <string>

#include "GridayTypes.hpp"
#include "GridayException.hpp"

#include "Vector.hpp"
#include "Cell.hpp"
#include "AtomTypeMap.hpp"
#include "PairEnergy.hpp"

class ForceField
    {
public:
    using PairEnergies = std::vector< std::unique_ptr<PairEnergy> >;

    ForceField(const AtomTypeMap& map, const Cell& box = Cell {});
    ForceField(const ForceField& other);
    ForceField& operator = (const ForceField& other);
    ~ForceField() = default;

    void read(std::string filename);
    void print();

    void setSimulationBox(const Cell& box);

    PairEnergy& getPairEnergy(int i, int j);
    GReal getMaxRcut();
private:
    PairEnergies mPairEnergies;
    AtomTypeMap mAtomTypeMap;
    Cell mBox;
    };
