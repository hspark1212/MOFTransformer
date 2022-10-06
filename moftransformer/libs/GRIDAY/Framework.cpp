#include "Framework.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sstream>
#include <iomanip>

Framework::Framework(AtomTypeMap& typeMap) :
    mAtomTypeMap {typeMap}
    {
    //int n = mAtomTypeMap.getNumTypes();

    //mAtomPositions.resize(n);
    //mAtomIndices.resize(n);
    }

void
Framework::update(Cell cell, std::vector<Vectors> AtomPositions, std::vector< std::vector<int> > AtomIndices)
    {
    mCell = cell;
    mInvCell = inverse(cell);
    mAtomPositions = AtomPositions;
    mAtomIndices   = AtomIndices;
    }

void
Framework::read(std::string filename)
    {
    // Read cssr file
    using namespace std;

    const GReal PI = 3.141592;

    FILE* fp = fopen(filename.c_str(), "r");
    if (fp == nullptr)
        THROW_EXCEPT("Cannot open cssr file.");

    char buffer[256];

    for (int i = 0; i < 3; ++i)
        {
        fscanf(fp, "%s", buffer);
        mCellLength[i] = atof(buffer);
        }

    for (int i = 0; i < 3; ++i)
        {
        fscanf(fp, "%s", buffer);
        mCellAngle[i] = atof(buffer) / 180.0 * PI;
        }

    this->makeCellInformation();

    fscanf(fp, "%[^\n]", buffer);

    fscanf(fp, "%s", buffer);
    int numAtoms = atoi(buffer);

    fscanf(fp, "%[^\n]", buffer);
    fscanf(fp, "%s", buffer);
    fscanf(fp, "%[^\n]", buffer);

    // Reset containers
    {
    int n = mAtomTypeMap.getNumTypes();

    mAtomPositions.clear();
    mAtomPositions.resize(n);

    mAtomIndices.clear();
    mAtomIndices.resize(n);
    }

    for (int i = 0; i < numAtoms; ++i)
        {
        fscanf(fp, "%s", buffer);
        int index = atoi(buffer);

        fscanf(fp, "%s", buffer);
        string type {buffer};

        Vector f;

        for (int j = 0; j < 3; ++j)
            {
            fscanf(fp, "%s", buffer);
            f[j] = atof(buffer);
            }

        Vector position = mCell * f;

        int typeIndex = mAtomTypeMap.getIndex(type);

        mAtomIndices[typeIndex].push_back(index);
        mAtomPositions[typeIndex].push_back(position);

        fscanf(fp, "%[^\n]", buffer);
        }
    }


void
Framework::update_all(Vector cellLength, Vector cellAngle, std::vector<Vectors> AtomPositions, std::vector<std::vector<int>> AtomIndices)
    {
    //const GReal PI = 3.141592;
    mCellLength = cellLength;
    mCellAngle  = cellAngle;
    this->makeCellInformation();

    mAtomPositions = AtomPositions;
    mAtomIndices   = AtomIndices;
    }


void
Framework::print()
    {
    using namespace std;

    const GReal PI = 3.141592;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(20) << "Cell length:" <<
            setw(10) << "a =" << setw(10) << mCellLength[0] <<
            setw(10) << "b =" << setw(10) << mCellLength[1] <<
            setw(10) << "c =" << setw(10) << mCellLength[2] <<
            endl;

    cout << setw(20) << "Cell  angle:" <<
            setw(10) << "alpha =" << setw(10) << mCellAngle[0] / PI * 180.0 <<
            setw(10) << "beta ="  << setw(10) << mCellAngle[1] / PI * 180.0 <<
            setw(10) << "gamma =" << setw(10) << mCellAngle[2] / PI * 180.0 <<
            endl;

    cout << setw(20) << "Cell height:" <<
            setw(10) << "bc =" << setw(10) << mCellHeights[0] <<
            setw(10) << "ca =" << setw(10) << mCellHeights[1] <<
            setw(10) << "ab =" << setw(10) << mCellHeights[2] <<
            endl << endl;

    cout << setw(20) << "Cell matrix:" << endl;
    cout << mCell << endl << endl;

    cout << setw(20) << "Volume:" << setw(20) << mVolume << endl << endl;

    cout << setw(20) << "# of atoms:" <<
            setw(15) << this->getNumAtoms() << endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(10) << "Index" << setw(10) << "Type" <<
            setw(15) << "Position" << endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    int n = mAtomTypeMap.getNumTypes();
    for (int i = 0; i < n; ++i)
        {
        string type = mAtomTypeMap.getName(i);

        int size = mAtomPositions[i].size();

        auto& pos = mAtomPositions[i];
        auto& idx = mAtomIndices[i];
        for (int j = 0; j < size; ++j)
            {
            cout << setw(10) << idx[j] <<
                    setw(10) << type <<
                    pos[j] <<
                    endl;
            }
        }
    }

void
Framework::expand(int nx, int ny, int nz)
    {
    int nn = nx * ny * nz;

    if (nn == 1)
        return;

    int size = mAtomTypeMap.getNumTypes();

    std::vector<int> sizes (size);
    for (int i = 0; i < size; ++i)
        sizes[i] = mAtomPositions[i].size();

    int ntot = this->getNumAtoms();
    int adder = ntot;
    for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
    for (int k = 0; k < nz; ++k)
        {
        // i == j == k == 0
        if (not (i or j or k))
            continue;

        for (int l = 0; l < size; ++l)
            {
            Vector r {(GReal)i, (GReal)j, (GReal)k};
            r = mCell * r;

            for (int ll = 0; ll < sizes[l]; ++ll)
                {
                auto& pos = mAtomPositions[l];
                auto& idx = mAtomIndices[l];

                pos.push_back(pos[ll] + r);
                idx.push_back(idx[ll] + adder);
                }

            }

        adder += ntot;
        }

    mCellLength[0] *= static_cast<GReal>(nx);
    mCellLength[1] *= static_cast<GReal>(ny);
    mCellLength[2] *= static_cast<GReal>(nz);

    this->makeCellInformation();
    }

void
Framework::autoExpand(GReal rcut)
    {
    int nx = std::ceil(2.0 * rcut / mCellHeights[0]);
    int ny = std::ceil(2.0 * rcut / mCellHeights[1]);
    int nz = std::ceil(2.0 * rcut / mCellHeights[2]);

    this->expand(nx, ny, nz);
    }

int
Framework::getNumAtoms()
    {
    int size = 0;

    for (auto& idx : mAtomIndices)
        size += idx.size();

    return size;
    }

const std::vector<Vectors>&
Framework::getAtomPositions()
    {
    return mAtomPositions;
    }

const std::vector< std::vector<int> >&
Framework::getAtomIndices()
    {
    return mAtomIndices;
    }

Vector
Framework::getCellLengths()
    {
    return mCellLength;
    }

Vector
Framework::getCellAngles()
    {
    return mCellAngle;
    }

Vector
Framework::getCellHeights()
    {
    return mCellHeights;
    }

Cell
Framework::getCell()
    {
    return mCell;
    }

void
Framework::makeCellInformation()
    {
    using namespace std;

    GReal a = mCellLength[0];
    GReal b = mCellLength[1];
    GReal c = mCellLength[2];

    GReal cosa = cos(mCellAngle[0]);
    GReal cosb = cos(mCellAngle[1]);
    GReal cosg = cos(mCellAngle[2]);

    GReal sing = sin(mCellAngle[2]);

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
