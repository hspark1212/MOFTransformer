#pragma once

#include <string>
#include <array>
#include <vector>
#include <tuple>

#include "GridayTypes.hpp"
#include "GridayException.hpp"
#include "Vector.hpp"
#include "Cell.hpp"
//#include "Tricubic.hpp"

class MaterialGrid
    {
public:
    MaterialGrid();
    MaterialGrid(std::string filename, std::string atomtype);

    void read(std::string filename, std::string atomtype);
    void print();

    GReal at(GIndex nx, GIndex ny, GIndex nz);
    GReal at(const GIndex3& n);

    GReal operator () (GIndex nx, GIndex ny, GIndex nz);
    GReal operator () (const GIndex3& n);

    GReal interpolate(GReal x, GReal y, GReal z);
    GReal interpolate(const Vector& r);

    Vector getCellLengths();
    Vector getCellAngles();
    Vector getCellHeights();

    GIndex getMaxNx();
    GIndex getMaxNy();
    GIndex getMaxNz();
    GIndex getNumGrids();

    Cell getCell();

    GReal getVolume();

    std::vector<GIndex3> getLocalMaximumIndices(GReal minEnergy);
    Vectors getLocalMaximumPositions(GReal minEnergy);

    std::vector<GIndex3> getLowEnergyIndices(GReal lowEnergy);
    std::vector<GIndex3> getHighEnergyIndices(GReal highEnergy);

    void transformToProbability(GReal temper);
    void transformToInverse();
    void transformToLog();
    void transformToPartialMap(GReal ecut = 5000.0);
    void transformToMinus();

    const std::vector<GReal>& data();

    std::vector<GIndex3> fillVolume(GIndex3 initial, GReal ecut);
    Vectors atomPositionByLocalMax(GReal ecut);
    Vectors atomPositionByFloodFillLow(GReal ecut, GIndex minSize = 2);
    Vectors atomPositionByFloodFill(GReal ecut, GIndex minSize = 2);
    GReal   getDistance(const Vector& p1, const Vector& p2);
    Vectors removeOverlap(const Vectors& pos, GReal rcut);

private:
    void makeCellInformation();
    GIndex3 getPBCindex(GIndex nx, GIndex ny, GIndex nz);
private:
    static const GReal PI;

    std::vector<GReal> mGrid;

    std::array<double, 3> mDoubleCellLengths;
    Vector mCellLengths;
    Vector mCellAngles;
    Vector mCellHeights;

    GIndex mMaxNx;
    GIndex mMaxNy;
    GIndex mMaxNz;

    GIndex mNumGrids;

    Cell mCell;
    Cell mInvCell;
    GReal mVolume;

    GReal mMinimumEnergy;
    GIndex3 mMinimumEnergyIndex3;

    // Test variables
    std::vector< std::vector<GIndex3> > mChannels;
    //TriCubicInterpolator mInterpolator;
    };
