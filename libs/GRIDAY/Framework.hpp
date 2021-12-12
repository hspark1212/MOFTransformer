#pragma once

#include <string>
#include <vector>

#include "GridayTypes.hpp"
#include "GridayException.hpp"
#include "Vector.hpp"
#include "Cell.hpp"
#include "AtomTypeMap.hpp"

class Framework
    {
public:
    Framework(AtomTypeMap& typeMap);
    ~Framework() = default;

    void read(std::string filename);
    void print();
    void update_all(Vector cellLength, Vector cellAngle, std::vector<Vectors> AtomPositions, std::vector<std::vector<int>> AtomIndices);
    void update(Cell cell, std::vector<Vectors> AtomPositions, std::vector< std::vector<int> > AtomIndices);

    void expand(int nx, int ny, int nz);
    void autoExpand(GReal rcut);

    int getNumAtoms();
    const std::vector<Vectors>& getAtomPositions();
    const std::vector< std::vector<int> >& getAtomIndices();

    Vector getCellLengths();
    Vector getCellAngles();
    Vector getCellHeights();

    Cell getCell();

private:
    AtomTypeMap mAtomTypeMap;

    GReal mVolume;

    Vector mCellLength;
    Vector mCellAngle;
    Vector mCellHeights;

    Cell mCell;
    Cell mInvCell;

    std::vector<Vectors> mAtomPositions;
    std::vector< std::vector<int> > mAtomIndices;

    void makeCellInformation();
    };
