#include "NlistMaker.hpp"

#include <cstdio>
#include <cmath>
#include <limits>
#include <queue>

#include <vector>
#include <fstream>
#include <iomanip>

NlistMaker::NlistMaker(const AtomTypeMap& typeMap,
                     const Framework& framework,
                     const ForceField& forceField) :
    mAtomTypeMap {typeMap},
    mFramework {framework},
    mForceField {forceField}
    {

    }

void
NlistMaker::setAtomTypeMap(const AtomTypeMap& typeMap)
    {
    mAtomTypeMap = typeMap;
    }

void
NlistMaker::setFramework(const Framework& framework)
    {
    mFramework = framework;
    }

void
NlistMaker::setForceField(const ForceField& forceField)
    {
    mForceField = forceField;
    }

void
NlistMaker::make(const GReal cutDis, const GIndex maxSize,
                std::string outputFileStem)
    {
    using namespace std;

    Framework originalFramework {mFramework};


    Framework expandedFramework {mFramework};
    expandedFramework.autoExpand(10.0);
    mForceField.setSimulationBox(expandedFramework.getCell());
    expandedFramework.print();

    Cell unitcell = expandedFramework.getCell();
    Cell invcell = ::inverse(unitcell);
    Vector unitcellLengths = expandedFramework.getCellLengths();

    // 1. Atom positions(xyz)
    int oxygenIndex = mAtomTypeMap.getIndex("O");
    auto oriAtoms = originalFramework.getAtomPositions()[oxygenIndex];
    auto& atoms = expandedFramework.getAtomPositions()[oxygenIndex];
    GIndex numOxygens = static_cast<GIndex>(oriAtoms.size());
    cout << "Num Oxygens : " << numOxygens << endl;
    cout << "Expanded : " << atoms.size() << endl;

    // 2. distance
    auto cutDisSq = cutDis * cutDis;
    std::vector<GReal> nlist {};
    for (GIndex i = 0; i < numOxygens; ++i)
        {
        std::priority_queue<GReal> q;
        for (GIndex j = 0; j < atoms.size(); ++j)
            {
            if (i == j)
                continue;

            auto s1 = atoms[i];
            //cout << s1 << endl;
            auto s2 = atoms[j];
            //cout << s2 << endl;
            auto s = invcell * (s1 - s2);
            for (auto& si : s)
                {
                if (si > 0.5)
                    si -= 1.0;
                if (si < -0.5)
                    si += 1.0;
                }

            auto r = unitcell * s;
            auto rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            //cout << rsq << endl;

            if (rsq > cutDisSq)
                continue;

            q.push(static_cast<GReal>(std::sqrt(rsq)));
            }

        for (GIndex k = 0; k < maxSize; ++k)
            {
            if (!q.empty())
                {
                nlist.push_back(q.top());
                q.pop();
                }
            else
                nlist.push_back(0.0);
            }

        while (!q.empty())
            q.pop();
        }

    GIndex dataSize = numOxygens * maxSize;

    string dataFileName = outputFileStem + ".nlist";
    FILE* dataFile = fopen(dataFileName.c_str(), "wb");
    if (dataFile == NULL)
        THROW_EXCEPT(".nlist file open fails at saving");

    GReal pOxygen = static_cast<GReal>(numOxygens);
    GReal pMaxSize = static_cast<GReal>(maxSize);

    fwrite(&pOxygen, 1, sizeof(GReal), dataFile);
    fwrite(&pMaxSize, 1, sizeof(GReal), dataFile);
    fwrite(nlist.data(), sizeof (GReal), dataSize, dataFile);
    fclose(dataFile);

    cout << "Writing done" << endl;
    }


void NlistMaker::writeDistanceHistogram(std::string filename,
                                        const GReal bin,
                                        const GReal min,
                                        const GReal max)
    {
    using namespace std;
    GIndex histSize = ceil((max - min) / bin);
    vector<GReal> histogram (histSize);

    Framework originalFramework {mFramework};

    Framework expandedFramework {mFramework};
    expandedFramework.autoExpand(10.0);
    mForceField.setSimulationBox(expandedFramework.getCell());
    expandedFramework.print();

    Cell unitcell = expandedFramework.getCell();
    Cell invcell = ::inverse(unitcell);
    Vector unitcellLengths = expandedFramework.getCellLengths();

    // 1. Atom positions(xyz)
    int oxygenIndex = mAtomTypeMap.getIndex("O");
    auto oriAtoms = originalFramework.getAtomPositions()[oxygenIndex];
    auto& atoms = expandedFramework.getAtomPositions()[oxygenIndex];
    GIndex numOxygens = static_cast<GIndex>(oriAtoms.size());

    // 2. distance
    GReal minSq = min * min;
    GReal maxSq = max * max;
    for (GIndex i = 0; i < numOxygens; ++i)
        {
        for (GIndex j = 0; j < atoms.size(); ++j)
            {
            if (i == j)
                continue;

            auto s1 = atoms[i];
            //cout << s1 << endl;
            auto s2 = atoms[j];
            //cout << s2 << endl;
            auto s = invcell * (s1 - s2);
            for (auto& si : s)
                {
                if (si > 0.5)
                    si -= 1.0;
                if (si < -0.5)
                    si += 1.0;
                }

            auto r = unitcell * s;
            auto rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

            if (rsq < minSq or rsq > maxSq)
                continue;

            GReal dist = static_cast<GReal>(sqrt(rsq));
            GIndex idx = floor((dist - min) / bin);
            histogram[idx] += 1.0;
            }
        }

    ofstream file(filename);
    cout << histSize << endl;
    for (GIndex idx = 0; idx < histSize; ++idx)
        {
        GReal x = static_cast<GReal>(idx) * bin + min;
        file << setw(15) << x << setw(15) << histogram[idx] << endl;
        }
    }
