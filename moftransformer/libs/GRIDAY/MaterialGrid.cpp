#include "MaterialGrid.hpp"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>

#include <cstdio>
#include <cmath>

#include "Random.hpp"

const GReal MaterialGrid::PI = 3.141592;

MaterialGrid::MaterialGrid()
    {

    }

MaterialGrid::MaterialGrid(std::string filename, std::string atomtype)
    {
    this->read(filename, atomtype);
    }

void
MaterialGrid::read(std::string filename, std::string atomtype)
    {
    using namespace std;

    auto pos = filename.rfind(".grid");

    // Extract file stem if file has extension.
    string fileStem;
    if (pos != string::npos)
        fileStem = filename.substr(0, pos);
    else
        fileStem = filename;

    ifstream gridFile ((fileStem + ".grid"));

    if (not gridFile.good())
        {
        stringstream msg;
        msg << "File open fails: " << (fileStem + ".grid");
        THROW_EXCEPT(msg.str());
        }

    string dummy;
    //while (gridFile.good())
    while (gridFile >> dummy)
        {
        //gridFile >> dummy;

        if (dummy == "CELL_PARAMETERS")
            {
            gridFile >> mDoubleCellLengths[0];
            gridFile >> mDoubleCellLengths[1];
            gridFile >> mDoubleCellLengths[2];
            for (GIndex i = 0; i < 3; ++i)
                mCellLengths[i] = static_cast<float>(mDoubleCellLengths[i]);
            }
        else if (dummy == "CELL_ANGLES")
            {
            gridFile >> mCellAngles;
            mCellAngles = mCellAngles / 180.0 * PI;
            }
        else if (dummy == "GRID_NUMBERS")
            {
            gridFile >> mMaxNx >> mMaxNy >> mMaxNz;
            mNumGrids = mMaxNx  * mMaxNy  * mMaxNz;
            }
        else
            {
            stringstream msg;
            msg << "Invalid option: " << dummy;
            THROW_EXCEPT(msg.str());
            }
        }

    gridFile.close();

    this->makeCellInformation();

    // Read .griddata
    FILE* gridDataFile = fopen((fileStem + ".griddata").c_str(), "rb");
    if (atomtype == "O")
        gridDataFile = fopen((fileStem + ".O").c_str(), "rb");
    else if (atomtype == "Si")
        gridDataFile = fopen((fileStem + ".si").c_str(), "rb");
    else
        {
        stringstream msg;
        msg << "File open fails: " << (fileStem + ".MaterialType");
        THROW_EXCEPT(msg.str());
        }

    if (gridDataFile == nullptr)
        {
        stringstream msg;
        msg << "File open fails: " << (fileStem + ".MaterialType");
        THROW_EXCEPT(msg.str());
        }

    mGrid.clear();
    mGrid.resize(mNumGrids);

    fread(mGrid.data(), sizeof (GReal), mNumGrids, gridDataFile);
    fclose(gridDataFile);

    mMinimumEnergy = numeric_limits<GReal>::max();

    for (GIndex nz = 0; nz < mMaxNz; ++nz)
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nx = 0; nx < mMaxNx; ++nx)
        {
        GReal e = (*this)(nx, ny, nz);

        if (e < mMinimumEnergy)
            {
            mMinimumEnergy = e;
            mMinimumEnergyIndex3 = {nx, ny, nz};
            }
        }

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
MaterialGrid::print()
    {
    using namespace std;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(20) << "Cell length:" <<
            setw(10) << "a =" << setw(10) << mCellLengths[0] <<
            setw(10) << "b =" << setw(10) << mCellLengths[1] <<
            setw(10) << "c =" << setw(10) << mCellLengths[2] <<
            endl;

    cout << setw(20) << "Cell  angle:" <<
            setw(10) << "alpha =" << setw(10) << mCellAngles[0] / PI * 180.0 <<
            setw(10) << "beta ="  << setw(10) << mCellAngles[1] / PI * 180.0 <<
            setw(10) << "gamma =" << setw(10) << mCellAngles[2] / PI * 180.0 <<
            endl;

    cout << setw(20) << "Cell height:" <<
            setw(10) << "bc =" << setw(10) << mCellHeights[0] <<
            setw(10) << "ca =" << setw(10) << mCellHeights[1] <<
            setw(10) << "ab =" << setw(10) << mCellHeights[2] <<
            endl << endl;

    cout << setw(20) << "Cell matrix:" << endl;
    cout << mCell << endl << endl;

    cout << setw(20) << "Volume:" << setw(15) << mVolume << endl << endl;
    cout << setw(20) << "Minimum energy:" << setw(15) << mMinimumEnergy <<
            endl << endl;

    cout << setw(20) << "# of grids:" <<
            setw(7) << "nx =" << setw(5) << mMaxNx <<
            setw(7) << "ny =" << setw(5) << mMaxNy <<
            setw(7) << "nz =" << setw(5) << mMaxNz <<
            setw(14) << "total =" << setw(10) << mNumGrids <<
            endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;
    }

GReal
MaterialGrid::at(GIndex nx, GIndex ny, GIndex nz)
    {
    while (nx < 0)
        nx += mMaxNx;
    while (ny < 0)
        ny += mMaxNy;
    while (nz < 0)
        nz += mMaxNz;
    while (nx >= mMaxNx)
        nx -= mMaxNx;
    while (ny >= mMaxNy)
        ny -= mMaxNy;
    while (nz >= mMaxNz)
        nz -= mMaxNz;

    GIndex index = nx + mMaxNx * (ny + mMaxNy * nz);

    return mGrid[index];
    }

GReal
MaterialGrid::at(const GIndex3& n)
    {
    return this->at(n[0], n[1], n[2]);
    }

GReal
MaterialGrid::operator () (GIndex nx, GIndex ny, GIndex nz)
    {
    GIndex index = nx + mMaxNx * (ny + mMaxNy * nz);

    return mGrid[index];
    }

GReal
MaterialGrid::operator () (const GIndex3& n)
    {
    return (*this)(n[0], n[1], n[2]);
    }

GReal
MaterialGrid::interpolate(GReal x, GReal y, GReal z)
    {
    return this->interpolate(Vector {x, y, z});
    }


//GReal
//MaterialGrid::interpolate(const Vector& r)
//    {
//    return mInterpolator.interpolate(r);
//    }

Vector
MaterialGrid::getCellLengths()
    {
    return mCellLengths;
    }

Vector
MaterialGrid::getCellAngles()
    {
    return mCellAngles;
    }

Vector
MaterialGrid::getCellHeights()
    {
    return mCellHeights;
    }

GIndex
MaterialGrid::getMaxNx()
    {
    return mMaxNx;
    }

GIndex
MaterialGrid::getMaxNy()
    {
    return mMaxNy;
    }

GIndex
MaterialGrid::getMaxNz()
    {
    return mMaxNz;
    }

GIndex
MaterialGrid::getNumGrids()
    {
    return mNumGrids;
    }

Cell
MaterialGrid::getCell()
    {
    return mCell;
    }

GReal
MaterialGrid::getVolume()
    {
    return mVolume;
    }

std::vector<GIndex3>
MaterialGrid::getLocalMaximumIndices(GReal minEnergy)
    {
    using namespace std;

    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    vector<GIndex3> maxIndices;

    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        GIndex up   = nz + 1;
        if (up == maxNz)
            up = 0;

        GIndex down = nz - 1;
        if (nz == 0)
            down = maxNz - 1;

        GIndex right = ny + 1;
        if (right == maxNy)
            right = 0;

        GIndex left  = ny - 1;
        if (ny == 0)
            left = maxNy - 1;

        GIndex front = nx + 1;
        if (front == maxNx)
            front = 0;

        GIndex back  = nx - 1;
        if (nx == 0)
            back = maxNx - 1;

        GReal upValue    = (*this)(nx, ny, up);
        GReal downValue  = (*this)(nx, ny, down);

        GReal rightValue = (*this)(nx, right, nz);
        GReal leftValue  = (*this)(nx, left, nz);

        GReal frontValue = (*this)(front, ny, nz);
        GReal backValue  = (*this)(back, ny, nz);

        GReal currentValue = (*this)(nx, ny, nz);

        if (currentValue < minEnergy)
            continue;

        bool isMaximum =
            currentValue > upValue    and
            currentValue > downValue  and
            currentValue > rightValue and
            currentValue > leftValue  and
            currentValue > frontValue and
            currentValue > backValue;

        if (isMaximum)
            {
            maxIndices.push_back(GIndex3 {nx, ny, nz});
            }
        }

    return maxIndices;
    }

Vectors
MaterialGrid::getLocalMaximumPositions(GReal minEnergy)
    {
    using namespace std;

    vector<GIndex3> idx = this->getLocalMaximumIndices(minEnergy);
    Vectors pos (idx.size());

    int size = pos.size();
    for (int i = 0; i < size; ++i)
        {
        Vector r = {static_cast<GReal>(idx[i][0]) / mMaxNx,
                    static_cast<GReal>(idx[i][1]) / mMaxNy,
                    static_cast<GReal>(idx[i][2]) / mMaxNz};

        r = mCell * r;

        pos[i] = r;
        }

    return pos;
    }

std::vector<GIndex3>
MaterialGrid::getLowEnergyIndices(GReal lowEnergy)
    {
    using namespace std;

    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    vector<GIndex3> lowIndices;

    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        GReal currentValue = (*this)(nx, ny, nz);
        if (currentValue > lowEnergy)
            continue;
        else
            lowIndices.push_back(GIndex3 {nx, ny, nz});
        }

    return lowIndices;
    }

std::vector<GIndex3>
MaterialGrid::getHighEnergyIndices(GReal highEnergy)
    {
    using namespace std;
    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    vector<GIndex3> highIndices;

    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        GReal currentValue = (*this)(nx, ny, nz);
        if (currentValue <= highEnergy)
            continue;
        else
            highIndices.push_back(GIndex3 {nx, ny, nz});
        }

    return highIndices;
    }

void
MaterialGrid::transformToProbability(GReal temper)
    {
    // Transform to boltzmann factor
    for (auto& e : mGrid)
        e = std::exp(-e / temper);

    // Normalize
    GReal sum = 0.0;
    for (auto& e : mGrid)
        sum += e;

    for (auto& e : mGrid)
        e /= sum;

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
MaterialGrid::transformToInverse()
    {
    for (auto& e : mGrid)
        e = 1.0 / (e - mMinimumEnergy + 1.0);

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
MaterialGrid::transformToLog()
    {
    for (auto& e : mGrid)
        e = std::log(e - mMinimumEnergy + 1.0);
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
MaterialGrid::transformToPartialMap(GReal ecut)
    {
    for (auto& e : mGrid)
        if (e >= ecut)
            e = ecut;
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
MaterialGrid::transformToMinus()
    {
    for (auto& e : mGrid)
        e = -e;
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

const std::vector<GReal>&
MaterialGrid::data()
    {
    return mGrid;
    }

void
MaterialGrid::makeCellInformation()
    {
    using namespace std;

    GReal a = mCellLengths[0];
    GReal b = mCellLengths[1];
    GReal c = mCellLengths[2];

    GReal cosa = cos(mCellAngles[0]);
    GReal cosb = cos(mCellAngles[1]);
    GReal cosg = cos(mCellAngles[2]);

    GReal sing = sin(mCellAngles[2]);

    GReal v = a * b * c *
        sqrt(1.0 + 2.0 * cosa * cosb * cosg -
             cosa * cosa - cosb * cosb - cosg * cosg);

    mCell.a[0] = a;
    mCell.a[1] = 0.0;
    mCell.a[2] = 0.0;

    mCell.b[0] = b * cosg;
    mCell.b[1] = b * sing;
    mCell.b[2] = 0.0;

    mCell.c[0] = c * cosb;
    mCell.c[1] = c * (cosa - cosb * cosg) / sing;
    mCell.c[2] = v / a / b / sing;

    mInvCell = inverse(mCell);

    mVolume = det(mCell);

    mCellHeights[0] = mVolume / norm(cross(mCell.b, mCell.c));
    mCellHeights[1] = mVolume / norm(cross(mCell.c, mCell.a));
    mCellHeights[2] = mVolume / norm(cross(mCell.a, mCell.b));
    }

GIndex3
MaterialGrid::getPBCindex(GIndex nx, GIndex ny, GIndex nz)
    {
    while (nx < 0)
        nx += mMaxNx;
    while (ny < 0)
        ny += mMaxNy;
    while (nz < 0)
        nz += mMaxNz;
    while (nx >= mMaxNx)
        nx -= mMaxNx;
    while (ny >= mMaxNy)
        ny -= mMaxNy;
    while (nz >= mMaxNz)
        nz -= mMaxNz;

    GIndex3 pbc {nx, ny, nz};
    return pbc;
    }

std::vector<GIndex3>
MaterialGrid::fillVolume(GIndex3 initial, GReal ecut)
    {
    std::vector<GIndex3> returnList {};
    std::queue<GIndex3> floodQueue;

    floodQueue.push(initial);
    while (!floodQueue.empty())
        {
        GIndex3 target = floodQueue.front();
        floodQueue.pop();
        GIndex  index  = target[0] + mMaxNx * (target[1] + mMaxNy * target[2]);
        if (this->at(target) > ecut)
            {
            mGrid[index] = 0.0;
            returnList.push_back(target);
            }
        else
            continue;

        auto new_pos = this->getPBCindex(target[0] - 1, target[1], target[2]);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        new_pos = this->getPBCindex(target[0] + 1, target[1], target[2]);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        new_pos = this->getPBCindex(target[0], target[1] - 1, target[2]);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        new_pos = this->getPBCindex(target[0], target[1] + 1, target[2]);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        new_pos = this->getPBCindex(target[0], target[1], target[2] - 1);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        new_pos = this->getPBCindex(target[0], target[1], target[2] + 1);
        if (this->at(new_pos) > ecut)
            floodQueue.push(new_pos);
        }

    return returnList;
    }

Vectors
MaterialGrid::atomPositionByLocalMax(GReal ecut)
    {
    auto pos = this->getLocalMaximumPositions(ecut);

    // Positive, fractional coordinate
    for (auto& p : pos)
        {
        p = mInvCell * p;
        for (auto& si : p)
            {
            if (si > 1.0)
                si -= 1.0;
            if (si < 0.0)
                si += 1.0;
            }
        }

    return pos;
    }


Vectors
MaterialGrid::atomPositionByFloodFillLow(GReal ecut, GIndex minSize)
    {
    Vectors positionList {};
    auto saveGrid = mGrid;
    auto tempGrid = mGrid;
    GReal ecut_low = ecut - 0.1;
    GReal ecut_high = ecut;

    for (GIndex nx = 0; nx < mMaxNx; ++nx)
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(nx, ny, nz) > ecut_low)
            {
            // std::vector<GIndex3>
            auto lowPore = this->fillVolume(GIndex3 {nx, ny, nz},
                                         ecut_low);
            tempGrid = mGrid;
            mGrid = saveGrid;

            std::vector<std::vector<GIndex3>> highPoreList {};
            for (const auto& idx : lowPore)
                {
                if ((*this)(idx[0], idx[1], idx[2]) > ecut_high)
                    {
                    auto highPore = this->fillVolume(idx, ecut_high);
                    highPoreList.push_back(highPore);
                    }
                }
            mGrid = tempGrid;

            if (highPoreList.size() < 1)
                {
                highPoreList.push_back(lowPore);
                }

            for (const auto& pore : highPoreList)
                {
                if (pore.size() < minSize)
                    continue;

                Vector posSum {};
                Vector origin = {static_cast<GReal>(nx) / mMaxNx,
                                 static_cast<GReal>(ny) / mMaxNy,
                                 static_cast<GReal>(nz) / mMaxNz};
                GReal weightSum = static_cast<GReal>(0.0);

                for (const auto& idx : pore)
                    {
                    GIndex index = idx[0] + mMaxNx * (idx[1] + mMaxNy * idx[2]);
                    auto weight = static_cast<GReal>(saveGrid[index]);

                    Vector pos = {static_cast<GReal>(idx[0]) / mMaxNx,
                                  static_cast<GReal>(idx[1]) / mMaxNy,
                                  static_cast<GReal>(idx[2]) / mMaxNz};
                    //PBC
                    auto s = origin - pos;
                    for (GIndex i = 0; i < 3; ++i)
                        {
                        if (s[i] > 0.5)
                            pos[i] += 1.0;
                        if (s[i] < -0.5)
                            pos[i] -= 1.0;
                        }

                    //weighted sum
                    pos = pos * weight;
                    weightSum += weight;
                    posSum = posSum + pos;
                    }

                //GReal size = static_cast<GReal>(pore.size());
                Vector atomPos = posSum / weightSum;
                positionList.push_back(atomPos);
                }
            }
        }

    mGrid = saveGrid;

    for (auto& p : positionList)
        {
        for (auto& si : p)
            {
            if (si > 1.0)
                si -= 1.0;
            if (si < 0.0)
                si += 1.0;
            }
        }

    return positionList;
    }


Vectors
MaterialGrid::atomPositionByFloodFill(GReal ecut, GIndex minSize)
    {
    Vectors positionList {};
    auto saveGrid = mGrid;

    for (GIndex nx = 0; nx < mMaxNx; ++nx)
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(nx, ny, nz) > ecut)
            {
            // std::vector<GIndex3>
            auto pore = this->fillVolume(GIndex3 {nx, ny, nz},
                                         ecut);
            if (pore.size() < minSize)
                continue;

            Vector posSum {};
            Vector origin = {static_cast<GReal>(nx) / mMaxNx,
                             static_cast<GReal>(ny) / mMaxNy,
                             static_cast<GReal>(nz) / mMaxNz};
            GReal weightSum = static_cast<GReal>(0.0);

            for (const auto& idx : pore)
                {
                GIndex index = idx[0] + mMaxNx * (idx[1] + mMaxNy * idx[2]);
                auto weight = static_cast<GReal>(saveGrid[index]);

                Vector pos = {static_cast<GReal>(idx[0]) / mMaxNx,
                              static_cast<GReal>(idx[1]) / mMaxNy,
                              static_cast<GReal>(idx[2]) / mMaxNz};
                //PBC
                auto s = origin - pos;
                for (GIndex i = 0; i < 3; ++i)
                    {
                    if (s[i] > 0.5)
                        pos[i] += 1.0;
                    if (s[i] < -0.5)
                        pos[i] -= 1.0;
                    }

                //weighted sum
                pos = pos * weight;
                weightSum += weight;
                posSum = posSum + pos;
                }

            //GReal size = static_cast<GReal>(pore.size());
            Vector atomPos = posSum / weightSum;
            positionList.push_back(atomPos);
            }
        }

    mGrid = saveGrid;

    for (auto& p : positionList)
        {
        for (auto& si : p)
            {
            if (si > 1.0)
                si -= 1.0;
            if (si < 0.0)
                si += 1.0;
            }
        }

    return positionList;
    }


GReal
MaterialGrid::getDistance(const Vector& p1, const Vector& p2)
    {
    auto s = p1 - p2;
    for (auto& si : s)
        {
        if (si > 0.5)
            si -= 1.0;
        if (si < -0.5)
            si += 1.0;
        }
    auto p = mCell * s;
    GReal rsq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    GReal r = static_cast<GReal>(std::sqrt(rsq));
    return r;
    }

Vectors
MaterialGrid::removeOverlap(const Vectors& pos, GReal rcut)
    {
    Vectors newPos {};
    std::vector<GIndex> usedList {};
    for (GIndex i = 0; i < pos.size(); ++i)
        {
        if (std::find(usedList.begin(), usedList.end(), i) != usedList.end())
            continue;
        std::queue<GIndex> q;
        std::vector<GIndex> posList {};
        q.push(i);

        while(!q.empty())
            {
            GIndex target = q.front();
            q.pop();

            if (std::find(posList.begin(), posList.end(), target) != posList.end())
                continue;
            posList.push_back(target);
            usedList.push_back(target);

            for (GIndex j = 0; j < pos.size(); ++j)
                {
                if (std::find(posList.begin(), posList.end(), j) != posList.end())
                    continue;
                auto dis = this->getDistance(pos[target], pos[j]);
                if (dis < rcut)
                    q.push(j);
                }
            }

        Vector posSum {0.0, 0.0, 0.0};
        Vector origin = pos[posList[0]];
        for (const auto& k : posList)
            {
            auto target = pos[k];
            auto s = origin - target;
            for (GIndex idx = 0; idx < 3; ++idx)
                {
                if (s[idx] > 0.5)
                    target[idx] += 1.0;
                if (s[idx] < -0.5)
                    target[idx] -= 1.0;
                }

            posSum = posSum + target;
            }
        posSum = posSum / static_cast<GReal>(posList.size());
        newPos.push_back(posSum);
        }

    for (auto& p : newPos)
        {
        for (auto& si : p)
            {
            if (si > 1.0)
                si -= 1.0;
            if (si < 0.0)
                si += 1.0;
            }
        }
    return newPos;
    }
