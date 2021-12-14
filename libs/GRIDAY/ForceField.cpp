#include "ForceField.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>

#include "LennardJones.hpp"
#include "Gaussian.hpp"
#include "ShiftedLJ.hpp"

ForceField::ForceField(const AtomTypeMap& map, const Cell& box) :
    mAtomTypeMap {map},
    mBox {box}
    {

    }

ForceField::ForceField(const ForceField& other) :
    mAtomTypeMap {other.mAtomTypeMap},
    mBox {other.mBox}
    {
    int size = other.mPairEnergies.size();

    mPairEnergies.resize(size);
    for (int i = 0; i < size; ++i)
        mPairEnergies[i] = other.mPairEnergies[i]->clone();
    }

ForceField&
ForceField::operator = (const ForceField& other)
    {
    mAtomTypeMap = other.mAtomTypeMap;
    mBox = other.mBox;

    int size = other.mPairEnergies.size();

    mPairEnergies.resize(size);
    for (int i = 0; i < size; ++i)
        mPairEnergies[i] = other.mPairEnergies[i]->clone();

    return *this;
    }

void
ForceField::read(std::string filename)
    {
    std::fstream file {filename};

    if (not file)
        {
        std::stringstream msg;
        msg << "Opening force field file fails: " << filename;

        THROW_EXCEPT(msg.str());
        }

    int numTypes = 0;
    file >> numTypes;

    if (not file)
        THROW_EXCEPT("No number of types in force field file");

    int n = mAtomTypeMap.getNumTypes();

    mPairEnergies.resize(n * n);

    for (int t = 0; t < numTypes; ++t)
        {
        int dummy {};
        std::string name1 {};
        std::string name2 {};
        std::string type {};

        file >> dummy >> name1 >> name2 >> type;

        if (not file)
            THROW_EXCEPT("Invalid format in force field file");

        // Check atom type typo
        int i;
        int j;

        try {
            i = mAtomTypeMap.getIndex(name1);
            j = mAtomTypeMap.getIndex(name2);
            }
        catch (GridayException& e)
            {
            THROW_EXCEPT("Force field reading fails", e);
            }

        int index = i + n * j;

        // Check pair energy is defined previously
        if (mPairEnergies[index])
            {
            std::stringstream msg;
            msg << "Override pair energy definition: " <<
                   name1 << ", " << name2;

            THROW_EXCEPT(msg.str());
            }

        if (type == "LennardJones")
            {
            GReal eps;
            GReal sig;
            GReal rcut;

            file >> eps >> sig >> rcut;

            if (not file)
                {
                std::stringstream msg;
                msg << "Invalid epsilon or sigma value for LennardJones: " <<
                       name1 << ", " << name2;

                THROW_EXCEPT(msg.str());
                }

            mPairEnergies[index] =
                std::make_unique<LennardJones>(eps, sig, rcut);
            }
        else if (type == "Gaussian")
            {
            GReal scale;
            GReal width;
            GReal rcut;

            file >> scale >> width >> rcut;

            if (not file)
                {
                std::stringstream msg;
                msg << "Invalid scale or width value for Gaussian: " <<
                       name1 << ", " << name2;

                THROW_EXCEPT(msg.str());
                }
            mPairEnergies[index] =
                std::make_unique<Gaussian>(scale, width, rcut);
            }
        else if (type == "ShiftedLJ")
            {
            GReal eps;
            GReal sig;
            GReal rcut;

            file >> eps >> sig >> rcut;

            if (not file)
                {
                std::stringstream msg;
                msg << "Invalid epsilon or sigma value for ShiftedLJ: " <<
                       name1 << ", " << name2;

                THROW_EXCEPT(msg.str());
                }

            mPairEnergies[index] =
                std::make_unique<ShiftedLJ>(eps, sig, rcut);
            }
        else
            {
            std::stringstream msg;
            msg << "Unsupported pair energy: " << type;

            THROW_EXCEPT(msg.str());
            }
        }

    // Do pair energy mixing
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
        {
        int index = i + n * j;

        if (not mPairEnergies[index])
            {
            if (i == j)
                {
                std::stringstream msg;
                msg << "No pair energy information: " <<
                       mAtomTypeMap.getName(i) << " -- " <<
                       mAtomTypeMap.getName(j);
                THROW_EXCEPT(msg.str());
                }

            if (mPairEnergies[j + n * i])
                {
                mPairEnergies[index] = mPairEnergies[j + n * i]->clone();
                continue;
                }

            PairEnergy& e1 = *mPairEnergies[i + n * i];
            PairEnergy& e2 = *mPairEnergies[j + n * j];

            std::string name1 = e1.getName();
            std::string name2 = e2.getName();

            if (name1 != name2)
                {
                std::stringstream msg;
                msg << "Cannot mix different pair energy: " <<
                       name1 << ", " << name2;

                THROW_EXCEPT(msg.str());
                }

            if (name1 == "LennardJones")
                mPairEnergies[index] = mixPairEnergy<LennardJones>(e1, e2);
            if (name1 == "Gaussian")
                mPairEnergies[index] = mixPairEnergy<Gaussian>(e1, e2);
            if (name1 == "ShiftedLJ")
                mPairEnergies[index] = mixPairEnergy<ShiftedLJ>(e1, e2);
            }
        }

    this->setSimulationBox(mBox);
    }

void
ForceField::print()
    {
    using namespace std;

    int n = mAtomTypeMap.getNumTypes();

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << "Simulation box: " << endl << mBox << endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(10) << "Type1" <<
            setw(10) << "Type2" <<
            setw(10) << "Info"  << endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            {
            int index = i + n * j;
            cout << setw(10) << mAtomTypeMap.getName(i) <<
                    setw(10) << mAtomTypeMap.getName(j) <<
                    setw(6) << " ";

            if (mPairEnergies[index])
                mPairEnergies[index]->print();
            else
                cout << endl;
            }
    }

void
ForceField::setSimulationBox(const Cell& box)
    {
    mBox = box;

    for (auto& pairEnergy : mPairEnergies)
        pairEnergy->setSimulationBox(mBox);
    }

PairEnergy&
ForceField::getPairEnergy(int i, int j)
    {
    int n = mAtomTypeMap.getNumTypes();
    int index = i + n * j;

    return *mPairEnergies[index];
    }

GReal
ForceField::getMaxRcut()
    {
    int n = mAtomTypeMap.getNumTypes();
    GReal maxRcut = GReal {};

    for (int i = 0; i < n; ++i)
        {
        int index = i + n * i;

        GReal rcut = mPairEnergies[index]->getRcut();
        if (rcut > maxRcut)
            maxRcut = rcut;
        }

    return maxRcut;
    }
