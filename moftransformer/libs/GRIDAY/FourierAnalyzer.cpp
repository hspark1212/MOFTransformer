#include "FourierAnalyzer.hpp"

#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "GridayException.hpp"

FourierAnalyzer::FourierAnalyzer(EnergyGrid& grid) : mGrid {grid},
    mCoeff (1),
    mMaxKx {0},
    mMaxKy {0},
    mMaxKz {0}
    {
    mCoeff[0] = this->calculateCoeff(0, 0, 0);
    }

void
FourierAnalyzer::setMaxKs(GInt maxKx, GInt maxKy, GInt maxKz)
    {
    GIndex size = (2 * maxKx + 1) *
                  (2 * maxKy + 1) *
                  (2 * maxKz + 1);

    std::vector<GComplex> newCoeff (size);

    GIndex idx = 0;
    for (GInt kz = -maxKz; kz <= maxKz; ++kz)
    for (GInt ky = -maxKy; ky <= maxKy; ++ky)
    for (GInt kx = -maxKx; kx <= maxKx; ++kx)
        {
        if (kz <= mMaxKz and kz >= -mMaxKz and
            ky <= mMaxKy and ky >= -mMaxKy and
            kx <= mMaxKx and kx >= -mMaxKx)
            {
            newCoeff[idx] = this->coeff(kx, ky, kz);
            }
        else
            {
            newCoeff[idx] = this->calculateCoeff(kx, ky, kz);
            }

        idx++;
        }

    mCoeff = newCoeff;

    mMaxKx = maxKx;
    mMaxKy = maxKy;
    mMaxKz = maxKz;
    }

GComplex
FourierAnalyzer::calculateCoeff(GInt i, GInt j, GInt k)
    {
    using namespace constant;

    const std::vector<GReal>& grid = mGrid.data();

    GComplex coeff {0.0f};

    GIndex idx = 0;

    GIndex maxNx = mGrid.getMaxNx();
    GIndex maxNy = mGrid.getMaxNy();
    GIndex maxNz = mGrid.getMaxNz();

    GReal dsx = 1.0 / static_cast<GReal>(maxNx);
    GReal dsy = 1.0 / static_cast<GReal>(maxNy);
    GReal dsz = 1.0 / static_cast<GReal>(maxNz);

    GComplex TPI = static_cast<GReal>(2.0) * PI * I;

    GComplex X = std::exp(- TPI * (dsx * i));
    GComplex Y = std::exp(- TPI * (dsy * j));
    GComplex Z = std::exp(- TPI * (dsz * k));

    GComplex Xn = ONE;
    GComplex Yn = ONE;
    GComplex Zn = ONE;

    for (GIndex nz = 0; nz < maxNz; ++nz)
        {
        for (GIndex ny = 0; ny < maxNy; ++ny)
            {
            for (GIndex nx = 0; nx < maxNx; ++nx)
                {
                GReal e = grid[idx];

                coeff += e * Xn * Yn * Zn;

                idx++;

                Xn *= X;
                } // nx
            Xn = ONE;
            Yn *= Y;
            } // ny
        Yn = ONE;
        Zn *= Z;
        } // nz

    coeff *= dsx * dsy * dsz;

    return coeff;
    }

GComplex
FourierAnalyzer::coeff(GInt i, GInt j, GInt k)
    {
    i += mMaxKx;
    j += mMaxKy;
    k += mMaxKz;

    GIndex maxi = mMaxKx + mMaxKx + 1;
    GIndex maxj = mMaxKy + mMaxKy + 1;

    GIndex index = i + maxi * (j + maxj * k);

    return mCoeff[index];
    }


GComplex
FourierAnalyzer::at(const Vector& r)
    {
    using namespace constant;

    Vector s = inverse(mGrid.getCell()) * r;

    GComplex TPI = static_cast<GReal>(2.0) * PI * I;

    GComplex X1 = std::exp(TPI * s[0]);
    GComplex Y1 = std::exp(TPI * s[1]);
    GComplex Z1 = std::exp(TPI * s[2]);

    GComplex Xstart = std::exp(TPI * static_cast<GReal>(-mMaxKx) * s[0]);
    GComplex Ystart = std::exp(TPI * static_cast<GReal>(-mMaxKy) * s[1]);
    GComplex Zstart = std::exp(TPI * static_cast<GReal>(-mMaxKz) * s[2]);

    GComplex Xi = Xstart;
    GComplex Yj = Ystart;
    GComplex Zk = Zstart;

    GComplex sum {static_cast<GReal>(0.0),
                  static_cast<GReal>(0.0)};

    GIndex idx {0};

    for (GInt k = -mMaxKz; k <= mMaxKz; ++k)
        {
        for (GInt j = -mMaxKy; j <= mMaxKy; ++j)
            {
            for (GInt i = -mMaxKx; i <= mMaxKx; ++i)
                {
                sum += mCoeff[idx++] * Xi * Yj * Zk;

                Xi *= X1;
                }
            Xi = Xstart;
            Yj *= Y1;
            }
        Yj = Ystart;
        Zk *= Z1;
        }

    return sum;
    }

void
FourierAnalyzer::writeVisitInput(std::string fileStem, bool onlyInUnitCell)
    {
    using namespace std;

    Cell mCell = mGrid.getCell();
    Cell mInvCell = inverse(mCell);

    GIndex mMaxNx = mGrid.getMaxNx();
    GIndex mMaxNy = mGrid.getMaxNy();
    GIndex mMaxNz = mGrid.getMaxNz();

    Vector mCellLengths = mGrid.getCellLengths();

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

    GReal da = mCellLengths[0] / static_cast<GReal>(mMaxNx);
    GReal db = mCellLengths[1] / static_cast<GReal>(mMaxNy);
    GReal dc = mCellLengths[2] / static_cast<GReal>(mMaxNz);

    GIndex maxNx = ceil(norm(cell.a) / da);
    GIndex maxNy = ceil(norm(cell.b) / db);
    GIndex maxNz = ceil(norm(cell.c) / dc);

    GIndex numGrids = maxNx * maxNy * maxNz;

    Cell grid {cell.a / static_cast<GReal>(maxNx),
               cell.b / static_cast<GReal>(maxNy),
               cell.c / static_cast<GReal>(maxNz)};

    vector<GReal> energyGrid (numGrids);
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
                energyGrid[index] = this->at(r).real();
            }
        else
            {
            energyGrid[index] = this->at(r).real();
            }

        index++;
        /*
        r = grid * r;

        energyGrid[index] = this->at(r).real();

        index++;
        */
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
FourierAnalyzer::writeCoeff(std::string filename)
    {
    using namespace std;

    ofstream coeffFile {filename};

    if (not coeffFile.good())
        THROW_EXCEPT("File writing fails: " + filename);

    coeffFile << setw(10) << mMaxKx <<
                 setw(10) << mMaxKy <<
                 setw(10) << mMaxKz <<
                 endl;

    for (const auto& coeff : mCoeff)
        coeffFile << setw(15) << coeff.real() <<
                     setw(15) << coeff.imag() << endl;
    }

void
FourierAnalyzer::readCoeff(std::string filename)
    {
    std::ifstream coeffFile {filename};

    if (not coeffFile.good())
        THROW_EXCEPT("Bad coeff file: " + filename);

    GInt maxKx;
    GInt maxKy;
    GInt maxKz;
    if (not (coeffFile >> maxKx >> maxKy >> maxKz))
        THROW_EXCEPT("Invalid Kx, Ky, Kz: " + filename);

    GIndex size = (2 * maxKx + 1) *
                  (2 * maxKy + 1) *
                  (2 * maxKz + 1);

    std::vector<GComplex> newCoeff (size);

    GIndex idx = 0;
    GReal re;
    GReal im;
    for (GInt kz = -maxKz; kz <= maxKz; ++kz)
    for (GInt ky = -maxKy; ky <= maxKy; ++ky)
    for (GInt kx = -maxKx; kx <= maxKx; ++kx)
        {
        if (not (coeffFile >> re >> im))
            THROW_EXCEPT("Error in coeff file: " + filename);

        newCoeff[idx++] = GComplex {re, im};
        }

    mCoeff = newCoeff;

    mMaxKx = maxKx;
    mMaxKy = maxKy;
    mMaxKz = maxKz;
    }

GReal
FourierAnalyzer::henry(GReal temperature)
    {
    using namespace std;

    Cell mCell = mGrid.getCell();

    GIndex mMaxNx = mGrid.getMaxNx();
    GIndex mMaxNy = mGrid.getMaxNy();
    GIndex mMaxNz = mGrid.getMaxNz();

    Vector mCellLengths = mGrid.getCellLengths();

    Cell cell {mCell};

    GReal da = mCellLengths[0] / static_cast<GReal>(mMaxNx);
    GReal db = mCellLengths[1] / static_cast<GReal>(mMaxNy);
    GReal dc = mCellLengths[2] / static_cast<GReal>(mMaxNz);

    GIndex maxNx = ceil(norm(cell.a) / da);
    GIndex maxNy = ceil(norm(cell.b) / db);
    GIndex maxNz = ceil(norm(cell.c) / dc);

    Cell grid {cell.a / static_cast<GReal>(maxNx),
               cell.b / static_cast<GReal>(maxNy),
               cell.c / static_cast<GReal>(maxNz)};

    GReal sum = 0.0;
    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r = {static_cast<GReal>(nx),
                    static_cast<GReal>(ny),
                    static_cast<GReal>(nz)};

        r = grid * r;
        GReal e = this->at(r).real();

        sum += std::exp(-e / temperature);
        }

    sum /= static_cast<GReal>(maxNx * maxNy * maxNz);

    return sum;
    }

void
FourierAnalyzer::for_each(std::function<void(GReal e)> op)
    {
    using namespace std;

    Cell mCell = mGrid.getCell();

    GIndex mMaxNx = mGrid.getMaxNx();
    GIndex mMaxNy = mGrid.getMaxNy();
    GIndex mMaxNz = mGrid.getMaxNz();

    Vector mCellLengths = mGrid.getCellLengths();

    Cell cell {mCell};

    GReal da = mCellLengths[0] / static_cast<GReal>(mMaxNx);
    GReal db = mCellLengths[1] / static_cast<GReal>(mMaxNy);
    GReal dc = mCellLengths[2] / static_cast<GReal>(mMaxNz);

    GIndex maxNx = ceil(norm(cell.a) / da);
    GIndex maxNy = ceil(norm(cell.b) / db);
    GIndex maxNz = ceil(norm(cell.c) / dc);

    Cell grid {cell.a / static_cast<GReal>(maxNx),
               cell.b / static_cast<GReal>(maxNy),
               cell.c / static_cast<GReal>(maxNz)};

    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r = {static_cast<GReal>(nx),
                    static_cast<GReal>(ny),
                    static_cast<GReal>(nz)};

        r = grid * r;
        GReal e = this->at(r).real();

        op(e);
        }
    }

void
FourierAnalyzer::writeCoeffAsGrid(std::string filestem)
    {
    using namespace std;
    using namespace constant;

    ofstream gridFile {filestem + ".grid"};

    if (not gridFile.good())
        THROW_EXCEPT(".grid file open fails at saving.");

    Cell cell = mGrid.getCell();
    Cell reciprocalCell = 2.0 * PI * transpose(inverse(cell));

    GIndex maxNx = 2 * mMaxKx + 1;
    GIndex maxNy = 2 * mMaxKy + 1;
    GIndex maxNz = 2 * mMaxKz + 1;

    Vector a = reciprocalCell.a;
    Vector b = reciprocalCell.b;
    Vector c = reciprocalCell.c;

    cout << reciprocalCell << endl;

    GReal length0 = norm(a) * static_cast<GReal>(maxNx);
    GReal length1 = norm(b) * static_cast<GReal>(maxNy);
    GReal length2 = norm(c) * static_cast<GReal>(maxNz);

    GReal angle0 = acos(dot(b, c) / norm(b) / norm(c)) / PI * 180.0;
    GReal angle1 = acos(dot(c, a) / norm(c) / norm(a)) / PI * 180.0;
    GReal angle2 = acos(dot(a, b) / norm(a) / norm(b)) / PI * 180.0;

    gridFile << setw(20) << "CELL_PARAMETERS" <<
                setw(10) << length0 <<
                setw(10) << length1 <<
                setw(10) << length2 <<
                endl <<

                setw(20) << "CELL_ANGLES" <<
                setw(10) << angle0 <<
                setw(10) << angle1 <<
                setw(10) << angle2 <<
                endl <<

                setw(20) << "GRID_NUMBERS" <<
                setw(10) << maxNx <<
                setw(10) << maxNy <<
                setw(10) << maxNz;

    gridFile.close();

    FILE* gridDataFile = fopen((filestem + ".griddata").c_str(), "wb");
    if (gridDataFile == NULL)
        THROW_EXCEPT(".griddata file open fails at saving.");

    vector<GReal> data (mCoeff.size());
    for (GIndex i = 0; i < data.size(); ++i)
        data[i] = norm(mCoeff[i]);

    fwrite(data.data(), sizeof (GReal), maxNx * maxNy * maxNz, gridDataFile);
    fclose(gridDataFile);
    }
