#include "GridMaker.hpp"

#include <cstdio>
#include <cmath>
#include <limits>

#include <vector>
#include <fstream>
#include <iomanip>

GridMaker::GridMaker(const AtomTypeMap& typeMap,
                     const Framework& framework,
                     const ForceField& forceField) :
    mAtomTypeMap {typeMap},
    mFramework {framework},
    mForceField {forceField}
    {

    }

void
GridMaker::setAtomTypeMap(const AtomTypeMap& typeMap)
    {
    mAtomTypeMap = typeMap;
    }

void
GridMaker::setFramework(const Framework& framework)
    {
    mFramework = framework;
    }

void
GridMaker::setForceField(const ForceField& forceField)
    {
    mForceField = forceField;
    }

void
GridMaker::make(std::string guestName, const GReal spacing,
                std::string outputFileStem)
    {
    using namespace std;

    Framework expandedFramework {mFramework};
    expandedFramework.autoExpand(mForceField.getMaxRcut());
    mForceField.setSimulationBox(expandedFramework.getCell());
    expandedFramework.print();

    Cell unitcell = mFramework.getCell();
    Vector unitcellLengths = mFramework.getCellLengths();

    // Calculate the number of grid boxs
    // GIndex maxNx = round(unitcellLengths[0] / spacing);
    // GIndex maxNy = round(unitcellLengths[1] / spacing);
    // GIndex maxNz = round(unitcellLengths[2] / spacing);

    // For COTA
    GIndex maxNx = static_cast<GIndex>(unitcellLengths[0] / spacing);
    GIndex maxNy = static_cast<GIndex>(unitcellLengths[1] / spacing);
    GIndex maxNz = static_cast<GIndex>(unitcellLengths[2] / spacing);

    cout << "# of grids: " <<
            "nx = " << maxNx << ", " <<
            "ny = " << maxNy << ", " <<
            "nz = " << maxNz << endl << endl;

    GIndex numGrids = maxNx * maxNy * maxNz;
    GIndex printFreq = numGrids / 100;
    GIndex runCounter = 0;

    Cell gridBox;
    gridBox.a = unitcell.a / static_cast<GReal>(maxNx);
    gridBox.b = unitcell.b / static_cast<GReal>(maxNy);
    gridBox.c = unitcell.c / static_cast<GReal>(maxNz);

    cout << "grid a = " << gridBox.a << endl <<
            "grid b = " << gridBox.b << endl <<
            "grid c = " << gridBox.c << endl << endl;

    auto& atoms = expandedFramework.getAtomPositions();
    int guestIndex = mAtomTypeMap.getIndex(guestName);
    cout << "Guest index = " << guestIndex << endl << endl;

    cout << "Generating Energy grid..." << endl;

    GIndex idx = 0;
    //unique_ptr<GReal[]> grid (new GReal[numGrids]);
    vector<GReal> grid (numGrids);

    // Output header
    cout << setw(105) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(10) << "Progress" <<
            setw(30) << "Current nx ny nz" <<
            setw(45) << "Current position" <<
            setw(20) << "Current energy"   <<
            endl;

    cout << setw(105) << setfill('=') << "" << setfill(' ') << endl;
    // Fill x first, y second then z last.
    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r {static_cast<GReal>(nx),
                  static_cast<GReal>(ny),
                  static_cast<GReal>(nz)};

        r = gridBox * r;

        Vectors pos {r};
        GReal energy = 0.0;

        int numTypes = atoms.size();
        for (int i = 0; i < numTypes; ++i)
            energy += mForceField.getPairEnergy(i, guestIndex).
                                  calculate(atoms[i], pos);

        if (isnan(energy))
            energy = std::numeric_limits<GReal>::max();

        grid[idx] = energy;
        idx++;

        if (runCounter % printFreq == 0)
            {
            cout << "... " << 
                setw(3)  << runCounter / printFreq << "  %" <<
                setw(10) << nx << setw(10) << ny << setw(10) << nz <<
                r <<
                setw(20) << energy << endl;
            }

        runCounter++;
        }

    cout << endl;
    cout << "Writing data..." << endl;

    string gridFileName = outputFileStem + ".grid";
    ofstream gridFile {gridFileName};

    if (not gridFile.good())
        THROW_EXCEPT(".gird file open fails at saving");

    const GReal PI = 3.141592;
    Vector angles = mFramework.getCellAngles() / PI * 180.0;
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

    cout << "Writing done" << endl;
    }


void
GridMaker::make(std::string guestName,
                const GIndex maxNx, const GIndex maxNy, const GIndex maxNz,
                std::string outputFileStem)
    {
    using namespace std;
    Framework expandedFramework {mFramework};
    expandedFramework.autoExpand(mForceField.getMaxRcut());
    mForceField.setSimulationBox(expandedFramework.getCell());
    expandedFramework.print();

    Cell unitcell = mFramework.getCell();
    Vector unitcellLengths = mFramework.getCellLengths();

    // Calculate the number of grid boxs
    //GIndex maxNx = round(unitcellLengths[0] / spacing);
    //GIndex maxNy = round(unitcellLengths[1] / spacing);
    //GIndex maxNz = round(unitcellLengths[2] / spacing);

    cout << "# of grids: " <<
            "nx = " << maxNx << ", " <<
            "ny = " << maxNy << ", " <<
            "nz = " << maxNz << endl << endl;

    GIndex numGrids = maxNx * maxNy * maxNz;
    GIndex printFreq = numGrids / 100;
    GIndex runCounter = 0;

    Cell gridBox;
    gridBox.a = unitcell.a / static_cast<GReal>(maxNx);
    gridBox.b = unitcell.b / static_cast<GReal>(maxNy);
    gridBox.c = unitcell.c / static_cast<GReal>(maxNz);

    cout << "grid a = " << gridBox.a << endl <<
            "grid b = " << gridBox.b << endl <<
            "grid c = " << gridBox.c << endl << endl;

    auto& atoms = expandedFramework.getAtomPositions();
    int guestIndex = mAtomTypeMap.getIndex(guestName);
    cout << "Guest index = " << guestIndex << endl << endl;

    cout << "Generating Energy grid..." << endl;

    GIndex idx = 0;
    //unique_ptr<GReal[]> grid (new GReal[numGrids]);
    vector<GReal> grid (numGrids);

    // Output header
    cout << setw(105) << setfill('=') << "" << setfill(' ') << endl;

    cout << setw(10) << "Progress" <<
            setw(30) << "Current nx ny nz" <<
            setw(45) << "Current position" <<
            setw(20) << "Current energy"   <<
            endl;

    cout << setw(105) << setfill('=') << "" << setfill(' ') << endl;

    // Fill x first, y second then z last.
    for (GIndex nz = 0; nz < maxNz; ++nz)
    for (GIndex ny = 0; ny < maxNy; ++ny)
    for (GIndex nx = 0; nx < maxNx; ++nx)
        {
        Vector r {static_cast<GReal>(nx),
                  static_cast<GReal>(ny),
                  static_cast<GReal>(nz)};

        r = gridBox * r;

        Vectors pos {r};
        GReal energy = 0.0;

        int numTypes = atoms.size();
        for (int i = 0; i < numTypes; ++i)
            energy += mForceField.getPairEnergy(i, guestIndex).
                                  calculate(atoms[i], pos);

        if (isnan(energy))
            energy = std::numeric_limits<GReal>::max();

        grid[idx] = energy;
        idx++;

        if (runCounter % printFreq == 0)
            {
            cout << "... " << 
                setw(3)  << runCounter / printFreq << "  %" <<
                setw(10) << nx << setw(10) << ny << setw(10) << nz <<
                r <<
                setw(20) << energy << endl;
            }

        runCounter++;
        }

    cout << endl;
    cout << "Writing data..." << endl;

    string gridFileName = outputFileStem + ".grid";
    ofstream gridFile {gridFileName};

    if (not gridFile.good())
        THROW_EXCEPT(".gird file open fails at saving");

    const GReal PI = 3.141592;
    Vector angles = mFramework.getCellAngles() / PI * 180.0;
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

    cout << "Writing done" << endl;

    }
