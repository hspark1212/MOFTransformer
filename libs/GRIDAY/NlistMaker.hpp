#pragma once

#include <string>

#include "GridayException.hpp"

#include "AtomTypeMap.hpp"
#include "Framework.hpp"
#include "ForceField.hpp"

#include "Vector.hpp"
#include "Cell.hpp"

class NlistMaker
    {
public:
    NlistMaker(const AtomTypeMap& typeMap,
              const Framework& framework,
              const ForceField& forceField);
    ~NlistMaker() = default;

    void setAtomTypeMap(const AtomTypeMap& typeMap);
    void setFramework(const Framework& framework);
    void setForceField(const ForceField& forceField);

    void make(const GReal cutDis, const GIndex maxSize,
              std::string outputFileStem);
    void writeDistanceHistogram(std::string filename, const GReal bin,
                                const GReal min, const GReal max);


private:
    AtomTypeMap mAtomTypeMap;
    Framework mFramework;
    // Reference, force field is non-copyable
    ForceField mForceField;
    };
