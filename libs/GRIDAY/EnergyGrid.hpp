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

class EnergyGrid
    {
public:
    EnergyGrid();
    EnergyGrid(std::string filename);

    void read(std::string filename);
    void print();
    void writeVisitInput(std::string fileStem,
                         bool onlyInUnitCell = true,
                         GReal da = 0.2,
                         GReal db = 0.2,
                         GReal dc = 0.2);
    void writeCotaGrid(std::string fileStem, double initSpacing, double cutoff, int expand_x, int expand_y, int expand_z);
    void writeEnergyHistogram(std::string filename,
                              const GReal bin,
                              const GReal max);

    GReal at(GIndex nx, GIndex ny, GIndex nz);
    GReal at(const GIndex3& n);

    GReal operator () (GIndex nx, GIndex ny, GIndex nz);
    GReal operator () (const GIndex3& n);
    //GReal periodicAt(GIndex nx, GIndex ny, GIndex nz);
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
    Cell getContainingBox(GReal alpha = 0.0, GReal beta = 0.0, GReal gamma = 0.0);
    GReal getVolume();
    GReal getMinimumEnergy();
    GIndex3 getMinimumEnergyIndex3();
    GReal getInaccessibleRangeRatio(GReal inf);
    std::vector<GIndex3> getLocalMinimumIndices(GReal maxEnergy);
    Vectors getLocalMinimumPositions(GReal maxEnergy);

    std::vector<GIndex3> getLowEnergyIndices(GReal lowEnergy);
    std::vector<GIndex3> getHighEnergyIndices(GReal highEnergy);

    void transformToProbability(GReal temper);
    void transformToInverse();
    void transformToLog();
    void transformToPartialMap(GReal ecut = 5000.0);
    void transformToMinus();
    void transformToLinSoft(GReal eShift = 4500.0, GReal eCut = 5000.0);
    void transformToLogSoft(GReal eShift = 4500.0, GReal eCut = 5000.0);

    const std::vector<GReal>& data();

    GReal henry(GReal temperature);
    GReal hoA(GReal temperature);
    GReal meanValue();

    void writeVolumetricImage(std::string filename, GIndex size, bool fillBox, Vector rotation, Vector translation);
	void save(std::string outputFileStem);
    void rescale(std::string outputFileStem, GReal spacing);

    //Channel
    void poreBlocking(GReal ecut);
    std::vector<std::vector<GIndex3>> getChannels(GReal ecut);
    std::vector<bool> isChannel(GIndex3 initial, GReal ecut, bool blocking = false);
    std::vector<GIndex3> floodFill(GIndex3 initial, GReal ecut, GIndex direction);

    GReal getSurfaceArea(GReal blockCutoff = 5000.0, bool blocking = false);
    GReal getDi(GReal blockCutoff = 5000.0, bool blocking = false);
    //std::vector<GReal> getDf(GReal blockCutoff = 5000.0);
    //std::vector<GReal> getDif(GReal blockCutoff = 5000.0); 

private:
    void makeCellInformation();
    GIndex3 getPBCindex(GIndex nx, GIndex ny, GIndex nz);
    std::tuple< std::vector<GIndex3>, std::vector<GIndex3> > getEnter(std::vector<GIndex3> pore, GIndex direction);
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
