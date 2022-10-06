#include "AtomTypeMap.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

AtomTypeMap::AtomTypeMap()
    {

    }

void
AtomTypeMap::read(std::string filename)
    {
    std::fstream file {filename};

    if (not file)
        {
        std::stringstream msg;
        msg << "Opening atom type file fails: " << filename;

        THROW_EXCEPT(msg.str());
        }

    int numTypes = 0;
    file >> numTypes;

    if (not file)
        THROW_EXCEPT("No number of types in atom type file");

    mCharges.resize(numTypes);

    for (int i = 0; i < numTypes; ++i)
        {
        int index {};
        std::string name {};
        GReal charge {};

        file >> index >> name >> charge;
        if (not file)
            THROW_EXCEPT("Invalid format in atom type file");

        if (index < 0)
            THROW_EXCEPT("Negative index in atom type");

        if (mNameToIndexMap.count(name) != 0)
            {
            std::stringstream msg;
            msg << "Multiple definition of atom name exists!: " << name;
            THROW_EXCEPT(msg.str());
            }

        mNameToIndexMap[name] = index;

        if (mIndexToNameMap.count(index) != 0)
            {
            std::stringstream msg;
            msg << "Multiple definition of atom index exists!: " << index;
            THROW_EXCEPT(msg.str());
            }

        mIndexToNameMap[index] = name;

        if (index >= numTypes)
            THROW_EXCEPT("Out of range in atom index");

        mCharges[index] = charge;
        }
    }

void
AtomTypeMap::print()
    {
    using namespace std;

    int size = mCharges.size();
    auto& nameMap = mIndexToNameMap;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(10) << "Index" <<
            setw(10) << "Name"  <<
            setw(10) << "Charge" << endl;

    cout << setw(80) << setfill('=') << "" << setfill(' ') << endl;
    for (int i = 0; i < size; ++i)
        {
        std::cout << std::setw(10) << i <<
                     std::setw(10) << nameMap[i] <<
                     std::setw(10) << mCharges[i] <<
                     std::endl;
        }
    }

int
AtomTypeMap::getIndex(std::string atomName)
    {
    if (mNameToIndexMap.count(atomName) == 0)
        {
        std::stringstream msg;
        msg << "Invalid atom name: " << atomName;

        THROW_EXCEPT(msg.str());
        }

    return mNameToIndexMap[atomName];
    }

std::string
AtomTypeMap::getName(int index)
    {
    if (mIndexToNameMap.count(index) == 0)
        {
        std::stringstream msg;
        msg << "Invalid atom index: " << index;

        THROW_EXCEPT(msg.str());
        }

    return mIndexToNameMap[index];
    }

int
AtomTypeMap::getNumTypes()
    {
    return mNameToIndexMap.size();
    }
