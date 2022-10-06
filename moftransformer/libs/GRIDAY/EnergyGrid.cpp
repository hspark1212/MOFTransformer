#include "EnergyGrid.hpp"

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

const GReal EnergyGrid::PI = 3.141592;

EnergyGrid::EnergyGrid()
    {

    }

EnergyGrid::EnergyGrid(std::string filename)
    {
    this->read(filename);
    }

void
EnergyGrid::read(std::string filename)
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
    if (gridDataFile == nullptr)
        gridDataFile = fopen((fileStem + ".voxel").c_str(), "rb");
        if (gridDataFile == nullptr)
            gridDataFile = fopen((fileStem + ".times").c_str(), "rb");
            if (gridDataFile == nullptr)
                {
                stringstream msg;
                msg << "File open fails: " << (fileStem + ".griddata");
                THROW_EXCEPT(msg.str());
                }

    mGrid.clear();
    mGrid.resize(mNumGrids);

    fread(mGrid.data(), sizeof (GReal), mNumGrids, gridDataFile);
    fclose(gridDataFile);

    //mMinimumEnergy = *min_element(mGrid.begin(), mGrid.end());
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
EnergyGrid::print()
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

void
EnergyGrid::writeVisitInput(std::string fileStem,
                            bool onlyInUnitCell,
                            GReal da,
                            GReal db,
                            GReal dc)
    {
    using namespace std;

    Vector a = mCell.a;
    Vector b = mCell.b;
    Vector c = mCell.c;

    Vectors vertices (8);
    vertices[0] = Vector {0, 0, 0};
    vertices[1] = a;
    vertices[2] = b;
    vertices[3] = c;
    vertices[4] = b + c;
    vertices[5] = c + a;
    vertices[6] = a + b;
    vertices[7] = a + b + c;

    const GReal max = numeric_limits<GReal>::max();
    const GReal min = numeric_limits<GReal>::min();

    GReal minX = max;
    GReal minY = max;
    GReal minZ = max;

    GReal maxX = min;
    GReal maxY = min;
    GReal maxZ = min;

    for (int i = 0; i < 8; ++i)
        {
        const auto& vertex = vertices[i];

        if (vertex[0] < minX)
            minX = vertex[0];
        if (vertex[0] > maxX)
            maxX = vertex[0];

        if (vertex[1] < minY)
            minY = vertex[1];
        if (vertex[1] > maxY)
            maxY = vertex[1];

        if (vertex[2] < minZ)
            minZ = vertex[2];
        if (vertex[2] > maxZ)
            maxZ = vertex[2];
        }

    Cell cell;
    cell.a = Vector {maxX - minX, 0, 0};
    cell.b = Vector {0, maxY - minY, 0};
    cell.c = Vector {0, 0, maxZ - minZ};

    //GReal da = mCellLengths[0] / static_cast<GReal>(mMaxNx);
    //GReal db = mCellLengths[1] / static_cast<GReal>(mMaxNy);
    //GReal dc = mCellLengths[2] / static_cast<GReal>(mMaxNz);

    GIndex maxNx = ceil(norm(cell.a) / da);
    GIndex maxNy = ceil(norm(cell.b) / db);
    GIndex maxNz = ceil(norm(cell.c) / dc);

    GIndex numGrids = maxNx * maxNy * maxNz;

    Cell grid {cell.a / static_cast<GReal>(maxNx),
               cell.b / static_cast<GReal>(maxNy),
               cell.c / static_cast<GReal>(maxNz)};

    vector<GReal> energyGrid (numGrids);

    Random random;

    GReal infinite = numeric_limits<GReal>::max();
    GIndex index = 0;
    Vector origin {minX, minY, minZ};
    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r = {static_cast<GReal>(nx),
                    static_cast<GReal>(ny),
                    static_cast<GReal>(nz)};

        r = grid * r + origin;

        if (onlyInUnitCell)
            {
            Vector s = mInvCell * r;

            bool isOut = false;
            for (const auto& si : s)
                {
                if (si > 1.0)
                    isOut = true;
                if (si < 0.0)
                    isOut = true;
                }

            if (isOut)
                energyGrid[index] = infinite;
            else
                energyGrid[index] = this->interpolate(r);
            }
        else
            {
            energyGrid[index] = this->interpolate(r);
            }

        index++;
        }

    ofstream bovFile {fileStem + ".bov"};
    if (not bovFile.good())
        THROW_EXCEPT("Writing VISIT input fails: bov open files");

    bovFile << "TIME: 1.000000" << endl <<
               "DATA_FILE: " << fileStem + ".times" << endl <<

               "DATA_SIZE:" << setw(8) << maxNx <<
                               setw(8) << maxNy <<
                               setw(8) << maxNz << endl <<

               "DATA_FORMAT: FLOAT"  << endl <<
               "VARIABLE: data"      << endl <<
               "DATA_ENDIAN: LITTLE" << endl <<
               "CENTERING: nodal"    << endl <<

               "BRICK_ORIGIN:" << setw(15) << minX <<
                                  setw(15) << minY <<
                                  setw(15) << minZ << endl <<

               "BRICK_SIZE:" << setw(15) << norm(cell.a) <<
                                setw(15) << norm(cell.b) <<
                                setw(15) << norm(cell.c);

    bovFile.close();

    FILE* timesFile = fopen((fileStem + ".times").c_str(), "wb");
    if (timesFile == nullptr)
        THROW_EXCEPT("Writing VISIT input filas: times open fails");

    fwrite(energyGrid.data(), sizeof (GReal), numGrids, timesFile);
    fclose(timesFile);
    }


void
EnergyGrid::writeCotaGrid(std::string fileStem, double initSpacing, double cutoff, int expand_x, int expand_y, int expand_z)
    {
    using namespace std;
    int maxl = 1000;
    int maxp = 10;
    float KELVIN_TO_J = 0.8314469;

    struct COTA_VECTOR
        {
        double x;
        double y;
        double z;
        };

    int nx = static_cast<int>(this->getMaxNx());
    int ny = static_cast<int>(this->getMaxNy());
    int nz = static_cast<int>(this->getMaxNz());

    FILE* cotaGridFile = fopen((fileStem + ".cotagrid").c_str(), "wb");
    if (cotaGridFile == nullptr)
        THROW_EXCEPT("Writing COTA GRID FAIL: cotagrid open fails");


    //1. spacing
    fwrite(&initSpacing, 1, sizeof(double), cotaGridFile);
    //2. grid size
    fwrite(&nx, 1, sizeof(int), cotaGridFile);
    fwrite(&ny, 1, sizeof(int), cotaGridFile);
    fwrite(&nz, 1, sizeof(int), cotaGridFile);
    //3. unitcell size
    double unitcell_x = static_cast<double>(mDoubleCellLengths[0]);
    double unitcell_y = static_cast<double>(mDoubleCellLengths[1]);
    double unitcell_z = static_cast<double>(mDoubleCellLengths[2]);
    fwrite(&unitcell_x, 1, sizeof(double), cotaGridFile);
    fwrite(&unitcell_y, 1, sizeof(double), cotaGridFile);
    fwrite(&unitcell_z, 1, sizeof(double), cotaGridFile);

    //4. Number of unitcell
    fwrite(&expand_x, 1, sizeof(int), cotaGridFile);
    fwrite(&expand_y, 1, sizeof(int), cotaGridFile);
    fwrite(&expand_z, 1, sizeof(int), cotaGridFile);
    //5. Cutoff distance
    fwrite(&cutoff, 1, sizeof(double), cotaGridFile);
    //6. maxl, maxp
    fwrite(&maxl, 1, sizeof(int), cotaGridFile);
    fwrite(&maxp, 1, sizeof(int), cotaGridFile);

    std::vector<COTA_VECTOR> VDWCoordinates {};
    //7. VDWCoordinates
    for (int i = 0; i < maxl + 2 * maxp + 2; ++i)
        {
        COTA_VECTOR a;
        a.x = 0.0;
        a.y = 0.0;
        a.z = 0.0;
        VDWCoordinates.push_back(a);
        }
    for (int i = -maxp; i < nx + maxp; ++i)
        VDWCoordinates[i + maxp].x = (unitcell_x / static_cast<double>(nx)) * static_cast<double>(i);
    for (int i = -maxp; i < ny + maxp; ++i)
        VDWCoordinates[i + maxp].y = (unitcell_y / static_cast<double>(ny)) * static_cast<double>(i);
    for (int i = -maxp; i < nz + maxp; ++i)
        VDWCoordinates[i + maxp].z = (unitcell_z / static_cast<double>(nz)) * static_cast<double>(i);

    for (int i = 0; i < maxl + 2 * maxp + 2; ++i)
        fwrite(&VDWCoordinates[i], 1, sizeof(VDWCoordinates[i]), cotaGridFile);

    //cout << nx << " " << ny << " " << nz << endl;
    std::vector<float> testGrid;
    testGrid.clear();
    testGrid.resize(nx * ny * nz);
    for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k)
        {
        int index = i + nx * (j + ny * k);
        testGrid[index] = mGrid[index] * KELVIN_TO_J;
        }

    fwrite(testGrid.data(), sizeof(float), nx * ny * nz, cotaGridFile);

    COTA_VECTOR a;
    a.x = 0.1;
    a.y = 0.1;
    a.z = 0.1;
    for (int i = 0; i < nx * ny * nz; ++i)
        fwrite(&a, 1, sizeof(COTA_VECTOR), cotaGridFile);

    fclose(cotaGridFile);
    }



void
EnergyGrid::writeEnergyHistogram(std::string filename,
                                 const GReal bin,
                                 const GReal max)
    {
    using namespace std;

    // Find minimum energy
    GReal min = mMinimumEnergy;

    GIndex histSize = ceil((max - min) / bin);
    vector<GReal> histogram (histSize);

    for (const auto& ene : mGrid)
        {
        if (ene > max)
            continue;

        GIndex idx = floor((ene - min) / bin);

        histogram[idx] += 1.0;
        }

    ofstream file (filename);
    for (GIndex idx = 0; idx < histSize; ++idx)
        {
        GReal x = static_cast<GReal>(idx) * bin + min;
        file << setw(15) << x << setw(15) << histogram[idx] << endl;
        }
    }


GReal
EnergyGrid::at(GIndex nx, GIndex ny, GIndex nz)
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

    /*
    if (nx > mMaxNx)
        THROW_EXCEPT("nx > mMaxNx");
    if (ny > mMaxNy)
        THROW_EXCEPT("ny > mMaxNy");
    if (nz > mMaxNz)
        THROW_EXCEPT("nz > mMaxNz");
    */

    /*
    if (nx < 0)
        THROW_EXCEPT("nx < 0");
    if (ny < 0)
        THROW_EXCEPT("ny < 0");
    if (nz < 0)
        THROW_EXCEPT("nz < 0");
    */

    GIndex index = nx + mMaxNx * (ny + mMaxNy * nz);

    return mGrid[index];
    }

GReal
EnergyGrid::at(const GIndex3& n)
    {
    return this->at(n[0], n[1], n[2]);
    }

GReal
EnergyGrid::operator () (GIndex nx, GIndex ny, GIndex nz)
    {
    GIndex index = nx + mMaxNx * (ny + mMaxNy * nz);

    return mGrid[index];
    }

GReal
EnergyGrid::operator () (const GIndex3& n)
    {
    return (*this)(n[0], n[1], n[2]);
    }
/*
GReal
EnergyGrid::periodicAt(GIndex nx, GIndex nz, GIndex nz)
    {

    }
*/

GReal
EnergyGrid::interpolate(GReal x, GReal y, GReal z)
    {
    return this->interpolate(Vector {x, y, z});
    }

/*
GReal
EnergyGrid::interpolate(const Vector& r)
    {
    return mInterpolator.interpolate(r);
    }
*/

GReal
EnergyGrid::interpolate(const Vector& r)
    {
    using namespace std;

    // Apply PBC
    Vector s = mInvCell * r;

    // Make s[i] to be in [0,1]
    for (auto& si : s)
        si = si - floor(si);

    vector<GReal> maxes {static_cast<GReal>(mMaxNx),
                         static_cast<GReal>(mMaxNy),
                         static_cast<GReal>(mMaxNz)};

    // Position at index space
    Vector rIndex = s;
    for (int i = 0; i < 3; ++i)
        rIndex[i] *= maxes[i];

    // Lower index
    vector<GIndex> idx0 (3);
    for (int i = 0; i < 3; ++i)
        idx0[i] = floor(rIndex[i]);

    // Upper index
    vector<GIndex> idx1 (3);
    for (int i = 0; i < 3; ++i)
        idx1[i] = idx0[i] + 1;

    if (idx1[0] == mMaxNx)
        idx1[0] = 0;
    if (idx1[1] == mMaxNy)
        idx1[1] = 0;
    if (idx1[2] == mMaxNz)
        idx1[2] = 0;

    // x, y, z in [0,1]
    GReal x = rIndex[0] - static_cast<GReal>(idx0[0]);
    GReal y = rIndex[1] - static_cast<GReal>(idx0[1]);
    GReal z = rIndex[2] - static_cast<GReal>(idx0[2]);

    GReal one_x = 1.0 - x;
    GReal one_y = 1.0 - y;
    GReal one_z = 1.0 - z;

    GReal v000 = (*this)(idx0[0], idx0[1], idx0[2]);
    GReal v100 = (*this)(idx1[0], idx0[1], idx0[2]);
    GReal v010 = (*this)(idx0[0], idx1[1], idx0[2]);
    GReal v001 = (*this)(idx0[0], idx0[1], idx1[2]);
    GReal v101 = (*this)(idx1[0], idx0[1], idx1[2]);
    GReal v011 = (*this)(idx0[0], idx1[1], idx1[2]);
    GReal v110 = (*this)(idx1[0], idx1[1], idx0[2]);
    GReal v111 = (*this)(idx1[0], idx1[1], idx1[2]);

    GReal ene =
         v000 * one_x * one_y * one_z +
         v100 *     x * one_y * one_z +
         v010 * one_x *     y * one_z +
         v001 * one_x * one_y *     z +
         v101 *     x * one_y *     z +
         v011 * one_x *     y *     z +
         v110 *     x *     y * one_z +
         v111 *     x *     y *     z;

    return ene;
    }


Vector
EnergyGrid::getCellLengths()
    {
    return mCellLengths;
    }

Vector
EnergyGrid::getCellAngles()
    {
    return mCellAngles;
    }

Vector
EnergyGrid::getCellHeights()
    {
    return mCellHeights;
    }

GIndex
EnergyGrid::getMaxNx()
    {
    return mMaxNx;
    }

GIndex
EnergyGrid::getMaxNy()
    {
    return mMaxNy;
    }

GIndex
EnergyGrid::getMaxNz()
    {
    return mMaxNz;
    }

GIndex
EnergyGrid::getNumGrids()
    {
    return mNumGrids;
    }

Cell
EnergyGrid::getCell()
    {
    return mCell;
    }

Cell
EnergyGrid::getContainingBox(GReal alpha, GReal beta, GReal gamma)
    {
    using namespace std;
    GReal one = static_cast<GReal>(1.0);
    GReal zero {};

    // matrix
    Cell Rx;
    Rx.a = {one, zero, zero};
    Rx.b = {zero, static_cast<GReal>(cos(alpha)), static_cast<GReal>(-sin(alpha))};
    Rx.c = {zero, static_cast<GReal>(sin(alpha)), static_cast<GReal>(cos(alpha))};

    Cell Ry;
    Ry.a = {static_cast<GReal>(cos(beta)), zero, static_cast<GReal>(-sin(beta))};
    Ry.b = {zero, one, zero};
    Ry.c = {static_cast<GReal>(sin(beta)), zero, static_cast<GReal>(cos(beta))};

    Cell Rz;
    Rz.a = {static_cast<GReal>(cos(gamma)), static_cast<GReal>(sin(gamma)), zero};
    Rz.b = {static_cast<GReal>(-sin(gamma)), static_cast<GReal>(cos(gamma)), zero};
    Rz.c = {zero, zero, one};

    auto rMatrix = Rx * Ry * Rz;

    Vector a = mCell.a;
    Vector b = mCell.b;
    Vector c = mCell.c;

    auto centor = (a + b + c) / 3.0;

    Vectors vertices (8);
    vertices[0] = Vector {0, 0, 0};
    vertices[1] = a;
    vertices[2] = b;
    vertices[3] = c;
    vertices[4] = b + c;
    vertices[5] = c + a;
    vertices[6] = a + b;
    vertices[7] = a + b + c;

    for (auto& x : vertices)
        x = rMatrix * (x - centor) + centor;

    const GReal max = numeric_limits<GReal>::max();
    const GReal min = numeric_limits<GReal>::min();

    GReal minX = max;
    GReal minY = max;
    GReal minZ = max;

    GReal maxX = min;
    GReal maxY = min;
    GReal maxZ = min;

    for (int i = 0; i < 8; ++i)
        {
        const auto& vertex = vertices[i];

        if (vertex[0] < minX)
            minX = vertex[0];
        if (vertex[0] > maxX)
            maxX = vertex[0];

        if (vertex[1] < minY)
            minY = vertex[1];
        if (vertex[1] > maxY)
            maxY = vertex[1];

        if (vertex[2] < minZ)
            minZ = vertex[2];
        if (vertex[2] > maxZ)
            maxZ = vertex[2];
        }

    Cell cell;
    cell.a = Vector {maxX - minX, 0, 0};
    cell.b = Vector {0, maxY - minY, 0};
    cell.c = Vector {0, 0, maxZ - minZ};

    return cell;
    }

GReal
EnergyGrid::getVolume()
    {
    return mVolume;
    }

GReal
EnergyGrid::getMinimumEnergy()
    {
    return mMinimumEnergy;
    }

GIndex3
EnergyGrid::getMinimumEnergyIndex3()
    {
    return mMinimumEnergyIndex3;
    }

GReal
EnergyGrid::getInaccessibleRangeRatio(GReal inf)
    {
    // inf = infinite energy.
    // so if E > inf, that region is inaccessible.
    using namespace std;

    GIndex numInf = count_if(mGrid.begin(), mGrid.end(),
        [inf](const GReal& v) {return v > inf;});

    return static_cast<GReal>(numInf) /
           static_cast<GReal>(mNumGrids);
    }

std::vector<GIndex3>
EnergyGrid::getLocalMinimumIndices(GReal maxEnergy)
    {
    using namespace std;
    // maxEnergy = 15kT in usual case

    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    //std::vector<GReal> mins;
    //Vectors minPositions;
    vector<GIndex3> minIndices;

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

        if (currentValue > maxEnergy)
            continue;

        bool isMinimum =
            currentValue < upValue    and
            currentValue < downValue  and
            currentValue < rightValue and
            currentValue < leftValue  and
            currentValue < frontValue and
            currentValue < backValue;

        if (isMinimum)
            {
            //mins.push_back(currentValue);
            //Vector r = {static_cast<GReal>(nx) / maxNx,
            //            static_cast<GReal>(ny) / maxNy,
            //            static_cast<GReal>(nz) / maxNz};

            //r = mCell * r;

            //minPositions.push_back(r);
            minIndices.push_back(GIndex3 {nx, ny, nz});
            }
        }

    //return minPositions;
    return minIndices;
    }

Vectors
EnergyGrid::getLocalMinimumPositions(GReal maxEnergy)
    {
    using namespace std;

    vector<GIndex3> idx = this->getLocalMinimumIndices(maxEnergy);
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
EnergyGrid::getLowEnergyIndices(GReal lowEnergy)
    {
    using namespace std;
    // lowEnergy = 15kT in usual case

    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    //std::vector<GReal> mins;
    //Vectors minPositions;
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
EnergyGrid::getHighEnergyIndices(GReal highEnergy)
    {
    using namespace std;
    // lowEnergy = 15kT in usual case

    GIndex maxNz = this->getMaxNz();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNx = this->getMaxNx();

    //std::vector<GReal> mins;
    //Vectors minPositions;
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
EnergyGrid::makeCellInformation()
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

void
EnergyGrid::transformToProbability(GReal temper)
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
EnergyGrid::transformToInverse()
    {
    for (auto& e : mGrid)
        e = 1.0 / (e - mMinimumEnergy + 1.0);

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
EnergyGrid::transformToLog()
    {
    for (auto& e : mGrid)
        e = std::log(e - mMinimumEnergy + 1.0);
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
EnergyGrid::transformToPartialMap(GReal ecut)
    {
    for (auto& e : mGrid)
        if (e >= ecut)
            e = ecut;
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
EnergyGrid::transformToMinus()
    {
    for (auto& e : mGrid)
        e = -e;
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
EnergyGrid::transformToLinSoft(GReal eShift, GReal eCut)
    {
    GReal eAtom = static_cast<GReal>(1e12);
    for (auto& e : mGrid)
        {
        if (e < eShift)
            {
            continue;
            }
        else if (e < eAtom)
            {
            e = (e - eShift) * (eCut - eShift) / eAtom + eShift;
            }
        else
            {
            e = eCut;
            }
        }

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }

void
EnergyGrid::transformToLogSoft(GReal eShift, GReal eCut)
    {
    GReal eAtom = static_cast<GReal>(1e12);
    for (auto& e : mGrid)
        {
        if (e < eShift)
            {
            continue;
            }
        else if (e < eAtom)
            {
            GReal alpha = (eCut - eShift) / std::log(eAtom / eShift);
            e = alpha * std::log(e / eShift) + eShift;
            }
        else
            {
            e = eCut;
            }
        }

    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }


const std::vector<GReal>&
EnergyGrid::data()
    {
    return mGrid;
    }


GReal
EnergyGrid::henry(GReal temperature)
    {
    GReal sum = 0;

    for (auto& e : mGrid)
        sum += std::exp(-e / temperature);

    sum /= static_cast<GReal>(mGrid.size());

    return sum;
    }


GReal
EnergyGrid::meanValue()
    {
    GReal sum = 0;

    for (const auto& e : mGrid)
        sum += e;

    sum /= static_cast<GReal>(mGrid.size());

    return sum;
    }


GReal
EnergyGrid::hoA(GReal temperature)
    {
    GReal hoA_sum = 0.0;
    GReal henry_sum = 0.0;

    for (auto& e : mGrid)
        {
        if (e < 75.0 * temperature)
            {
            henry_sum += std::exp(-e / temperature);
            hoA_sum += std::exp(-e / temperature) * e;
            }
        }

    henry_sum /= static_cast<GReal>(mGrid.size());
    hoA_sum /= static_cast<GReal>(mGrid.size());

    GReal sum = -(hoA_sum / henry_sum);
    sum += temperature;
    //sum *= temperature;
    return sum;
    }


void
EnergyGrid::poreBlocking(GReal ecut)
    {
    mChannels = {};
    auto saveGrid = mGrid;
    std::vector<GIndex3> saveList {};
    // x_zero enter
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(0, ny, nz) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {0, ny, nz}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {0, ny, nz}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // x_max
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(mMaxNx - 1, ny, nz) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {mMaxNx - 1, ny, nz}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {mMaxNx - 1, ny, nz}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // y_zero enter
    for (GIndex nx = 0; nx < mMaxNx; ++nx)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(nx, 0, nz) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {nx, 0, nz}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {nx, 0, nz}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // y_max
    for (GIndex nx = 0; nx < mMaxNx; ++nx)
    for (GIndex nz = 0; nz < mMaxNz; ++nz)
        {
        if ((*this)(nx, mMaxNy - 1, nz) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {nx, mMaxNy - 1, nz}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {nx, mMaxNy - 1, nz}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // z_zero enter
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nx = 0; nx < mMaxNx; ++nx)
        {
        if ((*this)(nx, ny, 0) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {nx, ny, 0}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {nx, ny, 0}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // z_max
    for (GIndex ny = 0; ny < mMaxNy; ++ny)
    for (GIndex nx = 0; nx < mMaxNx; ++nx)
        {
        if ((*this)(nx, ny, mMaxNz - 1) < ecut)
            {
            auto candidate  = this->isChannel(GIndex3 {nx, ny, mMaxNz - 1}, ecut, false);
            auto pore       = this->floodFill(GIndex3 {nx, ny, mMaxNz - 1}, ecut, -1);
            if (std::any_of(candidate.begin(), candidate.end(), [](bool i){return i;}))
                {
                mChannels.push_back(pore);
                for (const auto& x : pore)
                    saveList.push_back(x);
                }
            }
        }

    // others -> blocked
    for (GIndex nx = 1; nx < mMaxNx - 1; ++nx)
    for (GIndex ny = 1; ny < mMaxNy - 1; ++ny)
    for (GIndex nz = 1; nz < mMaxNz - 1; ++nz)
        {
        if ((*this)(nx, ny, nz) < ecut)
            {
            auto pore       = this->floodFill(GIndex3 {nx, ny, nz}, ecut, -1);
            }
        }

    // change save channels
    for (const auto& s : saveList)
        {
        GIndex  index  = s[0] + mMaxNx * (s[1] + mMaxNy * s[2]);
        mGrid[index]   = saveGrid[index];
        }
    //interpolator
    //mInterpolator.update(mCell, mMaxNx, mMaxNy, mMaxNz, mGrid);
    }


std::vector< std::vector<GIndex3> >
EnergyGrid::getChannels(GReal ecut)
    {
    auto saveGrid = mGrid;
    this->poreBlocking(ecut);
    mGrid = saveGrid;
    return mChannels;
    }


std::vector<bool>
EnergyGrid::isChannel(GIndex3 initial, GReal ecut, bool blocking)
    {
    std::vector<bool> dimension {false, false, false};
    auto saveGrid = mGrid;
    auto pore = this->floodFill(initial, ecut, -1);
    mGrid = saveGrid;

    std::vector<GIndex3> x_zeros {};
    std::vector<GIndex3> x_maxes {};
    std::tie(x_zeros, x_maxes) = this->getEnter(pore, 0);

    std::vector<GIndex3> y_zeros {};
    std::vector<GIndex3> y_maxes {};
    std::tie(y_zeros, y_maxes) = this->getEnter(pore, 2);

    std::vector<GIndex3> z_zeros {};
    std::vector<GIndex3> z_maxes {};
    std::tie(z_zeros, z_maxes) = this->getEnter(pore, 4);

    if (x_zeros.size() != 0)
        {
        auto channel = this->floodFill(x_zeros[0], ecut, 0);
        mGrid = saveGrid;

        std::vector<GIndex3> bound1 {};
        std::vector<GIndex3> bound2 {};
        std::tie(bound1, bound2) = this->getEnter(channel, 0);

        for (const auto& pos_one : bound1)
        for (const auto& pos_two : bound2)
            {
            if (pos_one[1] == pos_two[1] && pos_one[2] == pos_two[2])
                {
                dimension[0] = true;
                break;
                }
            }
        }

    if (y_zeros.size() != 0)
        {
        auto channel = this->floodFill(y_zeros[0], ecut, 2);
        mGrid = saveGrid;

        std::vector<GIndex3> bound1 {};
        std::vector<GIndex3> bound2 {};
        std::tie(bound1, bound2) = this->getEnter(channel, 2);

        for (const auto& pos_one : bound1)
        for (const auto& pos_two : bound2)
            {
            if (pos_one[0] == pos_two[0] && pos_one[2] == pos_two[2])
                {
                dimension[1] = true;
                break;
                }
            }
        }

    if (z_zeros.size() != 0)
        {
        auto channel = this->floodFill(z_zeros[0], ecut, 4);
        mGrid = saveGrid;

        std::vector<GIndex3> bound1 {};
        std::vector<GIndex3> bound2 {};
        std::tie(bound1, bound2) = this->getEnter(channel, 4);

        for (const auto& pos_one : bound1)
        for (const auto& pos_two : bound2)
            {
            if (pos_one[0] == pos_two[0] && pos_one[1] == pos_two[1])
                {
                dimension[2] = true;
                break;
                }
            }
        }

    if (blocking)
        {
        if (!(std::any_of(dimension.begin(), dimension.end(), [](bool i){return i;})))
            auto pore = this->floodFill(initial, ecut, -7);
        }

    return dimension;
    }


std::tuple< std::vector<GIndex3>, std::vector<GIndex3> >
EnergyGrid::getEnter(std::vector<GIndex3> pore, GIndex direction)
    {
    std::vector<GIndex3> enter_zero {};
    std::vector<GIndex3> enter_max  {};
    for (const auto& x : pore)
        {
        if (direction == 0 || direction == 1)
            {
            if (x[0] == 0)
                enter_zero.push_back(x);
            if (x[0] == mMaxNx - 1)
                enter_max.push_back(x);
            }
        if (direction == 2 || direction == 3)
            {
            if (x[1] == 0)
                enter_zero.push_back(x);
            if (x[1] == mMaxNy - 1)
                enter_max.push_back(x);
            }
        if (direction == 4 || direction == 5)
            {
            if (x[2] == 0)
                enter_zero.push_back(x);
            if (x[2] == mMaxNz - 1)
                enter_max.push_back(x);
            }
        }
    return std::make_tuple(enter_zero, enter_max);
    }


GIndex3
EnergyGrid::getPBCindex(GIndex nx, GIndex ny, GIndex nz)
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
EnergyGrid::floodFill(GIndex3 initial, GReal ecut, GIndex direction)
    {
    std::vector<GIndex3> returnList {};
    std::queue<GIndex3> floodQueue;

    floodQueue.push(initial);
    while (!floodQueue.empty())
        {
        GIndex3 target = floodQueue.front();
        floodQueue.pop();
        GIndex  index  = target[0] + mMaxNx * (target[1] + mMaxNy * target[2]);
        if (this->at(target) < ecut)
            {
            mGrid[index] = ecut;
            returnList.push_back(target);
            }
        else
            continue;

        // x
        auto new_pos = this->getPBCindex(target[0] - 1, target[1], target[2]);
        if (direction == 0)     // x_0 start
            {
            if (not (new_pos[0] == mMaxNx - 1))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }


        new_pos = this->getPBCindex(target[0] + 1, target[1], target[2]);
        if (direction == 1)     // x_max start
            {
            if (not (new_pos[0] == 0))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }

        // y
        new_pos = this->getPBCindex(target[0], target[1] - 1, target[2]);
        if (direction == 2)     // y_0 start
            {
            if (not (new_pos[1] == mMaxNy - 1))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }

        new_pos = this->getPBCindex(target[0], target[1] + 1, target[2]);
        if (direction == 3)     // y_max start
            {
            if (not (new_pos[1] == 0))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }

        // z
        new_pos = this->getPBCindex(target[0], target[1], target[2] - 1);
        if (direction == 4)     // z_0 start
            {
            if (not (new_pos[2] == mMaxNz - 1))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }

        new_pos = this->getPBCindex(target[0], target[1], target[2] + 1);
        if (direction == 5)     // z_max start
            {
            if (not (new_pos[2] == 0))
                {
                if (this->at(new_pos) < ecut)
                    floodQueue.push(new_pos);
                }
            }
        else
            {
            if (this->at(new_pos) < ecut)
                floodQueue.push(new_pos);
            }

        }
    return returnList;
    }



void
EnergyGrid::writeVolumetricImage(std::string filename,
                                  GIndex size,
                                  bool fillBox,
                                  Vector rotation,
                                  Vector translation)
    {
    // Hardcoding at this time.
    GIndex maxNx = size;
    GIndex maxNy = size;
    GIndex maxNz = size;

    using namespace std;
    GReal one = static_cast<GReal>(1.0);
    GReal zero {};

    auto alpha = rotation[0];
    auto beta  = rotation[1];
    auto gamma = rotation[2];

    // matrix
    Cell Rx;
    Rx.a = {one, zero, zero};
    Rx.b = {zero, static_cast<GReal>(cos(alpha)), static_cast<GReal>(-sin(alpha))};
    Rx.c = {zero, static_cast<GReal>(sin(alpha)), static_cast<GReal>(cos(alpha))};

    Cell Ry;
    Ry.a = {static_cast<GReal>(cos(beta)), zero, static_cast<GReal>(-sin(beta))};
    Ry.b = {zero, one, zero};
    Ry.c = {static_cast<GReal>(sin(beta)), zero, static_cast<GReal>(cos(beta))};

    Cell Rz;
    Rz.a = {static_cast<GReal>(cos(gamma)), static_cast<GReal>(sin(gamma)), zero};
    Rz.b = {static_cast<GReal>(-sin(gamma)), static_cast<GReal>(cos(gamma)), zero};
    Rz.c = {zero, zero, one};

    auto rMatrix = Rx * Ry * Rz;

    Vector a = mCell.a;
    Vector b = mCell.b;
    Vector c = mCell.c;

    Vectors vertices (8);
    vertices[0] = Vector {0, 0, 0};
    vertices[1] = a;
    vertices[2] = b;
    vertices[3] = c;
    vertices[4] = b + c;
    vertices[5] = c + a;
    vertices[6] = a + b;
    vertices[7] = a + b + c;

    const GReal max = numeric_limits<GReal>::max();
    const GReal min = numeric_limits<GReal>::min();

    GReal minX = max;
    GReal minY = max;
    GReal minZ = max;

    GReal maxX = min;
    GReal maxY = min;
    GReal maxZ = min;

    for (int i = 0; i < 8; ++i)
        {
        const auto& vertex = vertices[i];

        if (vertex[0] < minX)
            minX = vertex[0];
        if (vertex[0] > maxX)
            maxX = vertex[0];

        if (vertex[1] < minY)
            minY = vertex[1];
        if (vertex[1] > maxY)
            maxY = vertex[1];

        if (vertex[2] < minZ)
            minZ = vertex[2];
        if (vertex[2] > maxZ)
            maxZ = vertex[2];
        }

    GIndex numGrids = maxNx * maxNy * maxNz;

    Cell grid {{0.5, 0.0, 0.0},
               {0.0, 0.5, 0.0},
               {0.0, 0.0, 0.5}};

    vector<GReal> energyGrid (numGrids);

    Random random;

    GReal maxEnergy {static_cast<GReal>(5000.0)};
    GIndex index = 0;
    Vector origin {minX, minY, minZ};
    auto center = a + b + c;
    center = center / 3.0;
    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r = {static_cast<GReal>(nx),
                    static_cast<GReal>(ny),
                    static_cast<GReal>(nz)};

        r = grid * r;
        r = rMatrix * (r - center);
        r = r + origin + center + translation;

        Vector s = mInvCell * r;

        bool isOut = false;

        if (not fillBox)
            for (const auto& si : s)
                {
                if (si > 1.0)
                    isOut = true;
                if (si < 0.0)
                    isOut = true;
                }

        GReal energy {};

        if (isOut)
            energy = maxEnergy;
        else
            energy = this->interpolate(r);

        if (energy > maxEnergy)
            energy = maxEnergy;

        energyGrid[index] = energy;

        index++;
        }

    FILE* outfile = fopen(filename.c_str(), "wb");
    if (outfile == nullptr)
        THROW_EXCEPT("Writing VI fails: outfile open fails");

    fwrite(energyGrid.data(), sizeof (GReal), numGrids, outfile);
    fclose(outfile);
    }


GReal
EnergyGrid::getSurfaceArea(GReal blockCutoff, bool blocking)
    {
    // blocking
    if (blocking)
        this->poreBlocking(blockCutoff);

    // After blocking
    GReal  x_count = 0.0;
    GReal  y_count = 0.0;
    GReal  z_count = 0.0;
    GIndex maxNx = this->getMaxNx();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNz = this->getMaxNz();

    // Save surface grids
    for (GIndex nx = 0; nx < maxNx; ++nx)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nz = 0; nz < maxNz; ++nz)
        {
        GReal currentValue = (*this)(nx, ny, nz);
        if (currentValue < blockCutoff)
            {
            /*
            bool isBound =
                (*this)(nx + 1, ny, nz) >= blockCutoff or
                (*this)(nx - 1, ny, nz) >= blockCutoff or
                (*this)(nx, ny + 1, nz) >= blockCutoff or
                (*this)(nx, ny - 1, nz) >= blockCutoff or
                (*this)(nx, ny, nz + 1) >= blockCutoff or
                (*this)(nx, ny, nz - 1) >= blockCutoff;
            if (isBound)
                count += 1.0;
            */
            if (this->at(this->getPBCindex(nx + 1, ny, nz)) >= blockCutoff)
                x_count += 1.0;
            if (this->at(this->getPBCindex(nx - 1, ny, nz)) >= blockCutoff)
                x_count += 1.0;
            if (this->at(this->getPBCindex(nx, ny + 1, nz)) >= blockCutoff)
                y_count += 1.0;
            if (this->at(this->getPBCindex(nx, ny - 1, nz)) >= blockCutoff)
                y_count += 1.0;
            if (this->at(this->getPBCindex(nx, ny, nz + 1)) >= blockCutoff)
                z_count += 1.0;
            if (this->at(this->getPBCindex(nx, ny, nz - 1)) >= blockCutoff)
                z_count += 1.0;
            }
        }
    Cell cell = this->getCell();
    cell.a = cell.a / static_cast<GReal>(maxNx);
    cell.b = cell.b / static_cast<GReal>(maxNy);
    cell.c = cell.c / static_cast<GReal>(maxNz);

    //std::cout << x_count << " "  << y_count << " " << z_count << std::endl;
    GReal x_area = norm(cross(cell.b, cell.c)) * x_count;
    GReal y_area = norm(cross(cell.c, cell.a)) * y_count;
    GReal z_area = norm(cross(cell.a, cell.b)) * z_count;

    // conversion to cartesian coordinates
    return x_area + y_area + z_area;
    }


GReal
EnergyGrid::getDi(GReal blockCutoff, bool blocking)
    {
    if (blocking)
        this->poreBlocking(blockCutoff);

    GIndex maxNx = this->getMaxNx();
    GIndex maxNy = this->getMaxNy();
    GIndex maxNz = this->getMaxNz();

    std::vector<GIndex3> surface {};
    std::vector<GIndex3> inside  {};

    // Surface grids
    for (GIndex nx = 0; nx < maxNx; ++nx)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nz = 0; nz < maxNz; ++nz)
        {
        GReal currentValue = (*this)(nx, ny, nz);
        if (currentValue < blockCutoff)
            {
            inside.push_back(GIndex3 {nx, ny, nz});
            if ((*this)(nx + 1, ny, nz) >= blockCutoff)
                surface.push_back(GIndex3 {nx + 1, ny, nz});
            if ((*this)(nx - 1, ny, nz) >= blockCutoff)
                surface.push_back(GIndex3 {nx - 1, ny, nz});
            if ((*this)(nx, ny + 1, nz) >= blockCutoff)
                surface.push_back(GIndex3 {nx, ny + 1, nz});
            if ((*this)(nx, ny - 1, nz) >= blockCutoff)
                surface.push_back(GIndex3 {nx, ny - 1, nz});
            if ((*this)(nx, ny, nz + 1) >= blockCutoff)
                surface.push_back(GIndex3 {nx, ny, nz + 1});
            if ((*this)(nx, ny, nz - 1) >= blockCutoff)
                surface.push_back(GIndex3 {nx, ny, nz - 1});
            }
        }

    // calc min distance to surface
    GReal max_Di = std::numeric_limits<GReal>::min();
    for (const auto& pos_inside  : inside)
        {
        GReal min_distance = std::numeric_limits<GReal>::max();
        for (const auto& pos_surface : surface)
            {
            Vector r = {static_cast<GReal>((pos_inside[0] - pos_surface[0]) / mMaxNx),
                        static_cast<GReal>((pos_inside[1] - pos_surface[1]) / mMaxNy),
                        static_cast<GReal>((pos_inside[2] - pos_surface[2]) / mMaxNz)};

            r = mCell * r;
            auto dis = norm(r);
            if (dis < min_distance)
                min_distance = dis;
            }
        if (max_Di < min_distance)
            max_Di = min_distance;
        }

    return max_Di;
    }




/*
std::vector<GReal>
EnergyGrid::getDf(GReal blockCutoff)
    {
    for (const auto& channel : mChannels)
        {
        GIndex direction = -1;
        std::vector<bool> dimension = this->isChannel(channel[0], blockCutoff);
        if (dimension[2] == true)
            direction = 4;
        if (dimension[1] == true)
            direction = 2;
        if (dimension[0] == true)
            direction = 0;

        // get entrance
        std::vector<GIndex3> enter_zero, enter_max;
        std::tie(enter_zero, enter_max) = this->getEnter(channel, direction);

        // first probe
        std::vector<GIndex3> surface {};
        for (const auto& pos : enter_zero)
            {
            GReal currentValue = this->at(pos);
            if (currentValue < blockCutoff)
                {
                if ((*this)(pos[0] + 1, pos[1], pos[2]) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0] + 1, pos[1], pos[2]});
                if ((*this)(pos[0] - 1, pos[1], pos[2]) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0] - 1, pos[1], pos[2]});
                if ((*this)(pos[0], pos[1] + 1, pos[2]) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0], pos[1] + 1, pos[2]});
                if ((*this)(pos[0], pos[1] - 1, pos[2]) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0], pos[1] - 1, pos[2]});
                if ((*this)(pos[0], pos[1], pos[2] + 1) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0], pos[1], pos[2] + 1});
                if ((*this)(pos[0], pos[1], pos[2] - 1) >= blockCutoff)
                    surface.push_back(GIndex3 {pos[0], pos[1], pos[2] - 1});
                }
            }


        GIndex3 initial;
        GReal probe = std::numeric_limits<GReal>::min();
        for (const auto& pos_inside  : enter_zero)
            {
            GReal min_distance = std::numeric_limits<GReal>::max();
            for (const auto& pos_surface : surface)
                {
                Vector r = {static_cast<GReal>((pos_inside[0] - pos_surface[0]) / mMaxNx),
                            static_cast<GReal>((pos_inside[1] - pos_surface[1]) / mMaxNy),
                            static_cast<GReal>((pos_inside[2] - pos_surface[2]) / mMaxNz)};

                r = mCell * r;
                auto dis = norm(r);
                if (dis < min_distance)
                    min_distance = dis;
                }
            if (probe < min_distance)
                {
                probe = min_distance;
                initial = pos_inside;
                }
            }

        std::vector<GIndex3> returnList {};
        std::queue<GIndex3> floodQueue;

        floodQueue.push(initial);
        while (!floodQueue.empty())
            {
            GIndex3 target = floodQueue.front();
            floodQueue.pop();
            GIndex  index  = target[0] + mMaxNx * (target[1] + mMaxNy * target[2]);
            if (this->at(target) < ecut)
                {
                mGrid[index] = ecut;
                returnList.push_back(target);
                }
            else
                continue;
            }


        }

    }


std::vector<GReal>
EnergyGrid::getDif(GReal blockCutoff)
    {
    }

*/


void
EnergyGrid::save(std::string outputFileStem)
	{
	using namespace std;
	GIndex maxNx = this->getMaxNx();
	GIndex maxNy = this->getMaxNy();
	GIndex maxNz = this->getMaxNz();
	GIndex numGrids = maxNx * maxNy * maxNz;

	string gridFileName = outputFileStem + ".grid";
	ofstream gridFile {gridFileName};

	if (not gridFile.good())
        THROW_EXCEPT(".gird file open fails at saving");

	Vector unitcellLengths = this->getCellLengths();
	Vector angles		   = this->getCellAngles() / PI * 180.0;

    gridFile << setw(20) << "CELL_PARAMETERS" <<
                setw(10) << unitcellLengths[0] <<
                setw(10) << unitcellLengths[1] <<
                setw(10) << unitcellLengths[2] <<
                endl <<

                setw(20) << "CELL_ANGLES" <<
                setw(10) << angles[0] <<
                setw(10) << angles[1] <<
                setw(10) << angles[2] <<
                endl <<

                setw(20) << "GRID_NUMBERS" <<
                setw(10) << maxNx <<
                setw(10) << maxNy <<
                setw(10) << maxNz;

    gridFile.close();

    string gridDataFileName = outputFileStem + ".griddata";
    FILE* gridDataFile = fopen(gridDataFileName.c_str(), "wb");
    if (gridDataFile == NULL)
        THROW_EXCEPT(".griddata file open fails at saving");

    fwrite(mGrid.data(), sizeof (GReal), numGrids, gridDataFile);
    fclose(gridDataFile);
	}


void
EnergyGrid::rescale(std::string outputFileStem, GReal spacing)
    {
    using namespace std;

    auto unitcellLengths = this->getCellLengths();
    GIndex maxNx = static_cast<GIndex>(unitcellLengths[0] / spacing);
    GIndex maxNy = static_cast<GIndex>(unitcellLengths[1] / spacing);
    GIndex maxNz = static_cast<GIndex>(unitcellLengths[2] / spacing);

    GIndex numGrids = maxNx * maxNy * maxNz;

    Cell gridBox;
    gridBox.a = mCell.a / static_cast<GReal>(maxNx);
    gridBox.b = mCell.b / static_cast<GReal>(maxNy);
    gridBox.c = mCell.c / static_cast<GReal>(maxNz);

    GIndex idx = 0;
    vector<GReal> grid (numGrids);

    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r {static_cast<GReal>(nx),
                  static_cast<GReal>(ny),
                  static_cast<GReal>(nz)};

        r = gridBox * r;
        grid[idx] = this->interpolate(r);
        idx++;
        }

    string gridFileName = outputFileStem + ".grid";
    ofstream gridFile {gridFileName};

    if (not gridFile.good())
        THROW_EXCEPT(".gird file open fails at saving");

    const GReal PI = 3.141592;
    Vector angles = this->getCellAngles() / PI * 180.0;
    gridFile << setw(20) << "CELL_PARAMETERS" <<
                setw(10) << unitcellLengths[0] <<
                setw(10) << unitcellLengths[1] <<
                setw(10) << unitcellLengths[2] <<
                endl <<

                setw(20) << "CELL_ANGLES" <<
                setw(10) << angles[0] <<
                setw(10) << angles[1] <<
                setw(10) << angles[2] <<
                endl <<

                setw(20) << "GRID_NUMBERS" <<
                setw(10) << maxNx <<
                setw(10) << maxNy <<
                setw(10) << maxNz;

    gridFile.close();

    string gridDataFileName = outputFileStem + ".griddata";
    FILE* gridDataFile = fopen(gridDataFileName.c_str(), "wb");
    if (gridDataFile == NULL)
        THROW_EXCEPT(".griddata file open fails at saving");

    fwrite(grid.data(), sizeof (GReal), numGrids, gridDataFile);
    fclose(gridDataFile);
    }
