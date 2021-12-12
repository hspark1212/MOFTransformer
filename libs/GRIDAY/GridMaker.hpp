#pragma once

#include <string>

#include "GridayException.hpp"

#include "AtomTypeMap.hpp"
#include "Framework.hpp"
#include "ForceField.hpp"

#include "Vector.hpp"
#include "Cell.hpp"

class GridMaker
    {
public:
    GridMaker(const AtomTypeMap& typeMap,
              const Framework& framework,
              const ForceField& forceField);
    ~GridMaker() = default;

    void setAtomTypeMap(const AtomTypeMap& typeMap);
    void setFramework(const Framework& framework);
    void setForceField(const ForceField& forceField);

    void make(std::string guestName, const GReal spacing,
              std::string outputFileStem);
    void make(std::string guestName,
              const GIndex maxNx, const GIndex maxNy, const GIndex maxNz,
              std::string outputFileStem);

private:
    AtomTypeMap mAtomTypeMap;
    Framework mFramework;
    // Reference, force field is non-copyable
    ForceField mForceField;
    };
