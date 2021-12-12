#pragma once

#include <vector>
#include <string>
#include <functional>

#include "GridayTypes.hpp"
#include "Vector.hpp"
#include "Cell.hpp"
#include "EnergyGrid.hpp"

class FourierAnalyzer
    {
public:
    FourierAnalyzer(EnergyGrid& grid);
    void setMaxKs(GInt maxKx, GInt maxKy, GInt maxKz);
    GComplex coeff(GInt i, GInt j, GInt k);
    GComplex at(const Vector& r);

    void writeVisitInput(std::string fileStem, bool onlyInUnitCell = true);
    void writeCoeff(std::string filename);
    void readCoeff(std::string filename);
    void writeCoeffAsGrid(std::string filestem);

    GReal henry(GReal temperature);
    void for_each(std::function<void(GReal e)> op);
private:
    GComplex calculateCoeff(GInt i, GInt j, GInt k);
private:
    EnergyGrid& mGrid;

    std::vector<GComplex> mCoeff;

    GInt mMaxKx;
    GInt mMaxKy;
    GInt mMaxKz;
    };
