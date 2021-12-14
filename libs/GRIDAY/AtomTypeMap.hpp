#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "GridayTypes.hpp"
#include "GridayException.hpp"

class AtomTypeMap
    {
public:
    using NameToIndexMap = std::unordered_map<std::string, int>;
    using IndexToNameMap = std::unordered_map<int, std::string>;
    using Charges = std::vector<GReal>;

    AtomTypeMap();
    ~AtomTypeMap() = default;

    void read(std::string filename);
    void print();

    int getIndex(std::string atomName);
    std::string getName(int index);

    int getNumTypes();
private:
    NameToIndexMap mNameToIndexMap;
    IndexToNameMap mIndexToNameMap;
    Charges mCharges;
    };
