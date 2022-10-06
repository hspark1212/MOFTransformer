#include <string>
#include <iostream>
#include <iomanip>
#include <exception>
#include <regex>

#include "../Griday.hpp"
#include "../Vector.hpp"
#include "../Random.hpp"


std::vector<std::string>
split(const std::string& text, const std::string& delimiters = "\\s")
    {
    std::regex delimiter ("[" + delimiters + "]+");

    std::sregex_token_iterator endIter;
    std::sregex_token_iterator iter {text.begin(), text.end(), delimiter, -1};

    std::vector<std::string> tokens;
    for (;iter != endIter; ++iter)
        tokens.push_back(*iter);

    return tokens;
    }


int
main(int argc, char* argv[])
try {
    using namespace std;

    if (argc < 4)
        {
        cerr << "./sanity GRID TEMP ECUT (density)"
            << endl;
        return 1;
        }

    string grid_path    {argv[1]};
    GReal  temperature  {static_cast<GReal>(stof(argv[2]))};
    GReal  ecut         {static_cast<GReal>(stof(argv[3]))};
    GReal  density = 0.0;
    if (argc > 4)
        density =  {static_cast<GReal>(stof(argv[4]))};

    EnergyGrid grid;
    grid.read(grid_path);
    //grid.transformToPartialMap(5000.0);

    auto sList = split(grid_path, "/");
    auto st_name = sList[sList.size() - 1];

    // henry - henry(blocking)
    // void fraction - void (blocking)
    // Binding Energy - Binding Energy (blocking)
    // Channels - Channel Dimensions
    // Surface Area - Surface Area (blocking)
    // Di - Df - Dif
    density = 1e-30 * grid.getVolume();

    GReal henry_unit = grid.henry(temperature);
    GReal henry_den  = henry_unit / 8.314472 / temperature * density;

    GReal hoA_unit   = grid.hoA(temperature);
    GReal hoA_den    = hoA_unit * 8.314472 / 1000.0;

    GIndex nGrids    = grid.getMaxNx() * grid.getMaxNy() * grid.getMaxNz();
    GIndex nVoidGrids= grid.getLowEnergyIndices(ecut - 0.001).size();
    GReal void_frac  = nVoidGrids / static_cast<GReal>(nGrids);
    GReal sa         = grid.getSurfaceArea();
    GReal cell_vol   = grid.getVolume();
    GReal poreV      = void_frac * cell_vol;


    // after blocking
    grid.poreBlocking(15.0 * temperature);
    GReal henry_block       = grid.henry(temperature);
    GReal henry_block_den   = henry_block / 8.314472 / temperature * density;

    GReal hoA_block_unit    = grid.hoA(temperature);
    GReal hoA_block_den     = hoA_block_unit * 8.314472 / 1000.0;

    GIndex nVoidGrids_block = grid.getLowEnergyIndices(ecut - 0.001).size();
    GReal void_frac_block   = nVoidGrids_block / static_cast<GReal>(nGrids);
    GReal sa_block          = grid.getSurfaceArea();
    GReal cell_vol_block    = grid.getVolume();
    GReal poreV_block       = void_frac_block * cell_vol_block;

    auto channels  = grid.getChannels(ecut);

    // Print Session
    /*
    cout << "Henry Coefficient Blocking Ratio" << endl;
    //cout << henry_unit << "  \t" << "[]" << endl;
    cout << henry_unit / henry_block << endl << endl;

    cout << "Henry Coefficient w/o blocking" << endl;
    cout << henry_den << endl << endl;

    cout << "Henry Coefficient w blocking" << endl;
    cout << henry_block_den << endl << endl;

    cout << "HoA w/o blocking" << endl;
    cout << hoA_den << endl << endl;

    cout << "HoA w blocking" << endl;
    cout << hoA_block_den << endl << endl;

    cout << "Void Fraction w/o blocking" << endl;
    cout << void_frac << endl << endl;

    cout << "Void Fraction with blocking" << endl;
    cout << void_frac_block << endl << endl;

    cout << "SA w/o blocking [A^2]" << endl;
    cout << sa << endl << endl;

    cout << "SA with blocking [A^2]" << endl;
    cout << sa_block << endl << endl;

    cout << "Channels"  << endl;
    cout << channels.size() << endl;

    for (GIndex i = 0; i < channels.size(); ++i)
        {
        auto boolList = grid.isChannel(channels[i][0], ecut);
        //cout << channels[i].size();
        for (const auto& x : boolList)
            cout << x;
        cout << " ";
        }
    cout << endl << endl;
    */
    /*
    cout << "Henry Coefficient (blocking, 5000K)" << endl;
    cout << henry_block << "  \t" << "[]" << endl;
    if (density > 0.0)
        cout << henry_block / density * 1e-27 << "  \t" << "[moles/kg/Pa]" << endl;
    cout << endl;
    */

    cout.setf(ios::right);
    /*
    cout << setw(10) << "framework"     << "  "
         << setw(5)  << "gas"           << "  "
         << setw(24) << "KH(unit, moles/kg/Pa)" << "  "
         << setw(28) << "KH_block(unit, moles/kg/Pa)" << "  "
         << setw(12) << "KH_ratio"      << "  "
         << setw(25) << "HoA(w/o, w block)[kJ/mol]" << "  "
         << setw(24) << "Void (w/o, w block)" << "  "
         << setw(22) << "SA (w/o block, block)" << "  "
         << setw(10) << "Channels" << "  "
         << setw(10) << "Dimension" << "  "
         << endl;
    */
    cout << setw(10) << st_name  << "  "
         << setw(5)  << "CH4"           << "  "
         << setw(11) << henry_unit << setw(13)  << henry_den << "  "
         << setw(14) << henry_block << setw(14) << henry_block_den << "  "
         << setw(12) << henry_den / henry_block_den << "  "
         << setw(13) << hoA_den << setw(12) << hoA_block_den << "  "
         << setw(12) << void_frac << setw(12) << void_frac_block << "  "
         << setw(10) << cell_vol  << setw(10) << cell_vol_block  << "  "
         << setw(10) << poreV     << setw(10) << poreV_block     << "  "
         << setw(10) << channels.size() << "  ";

    for (GIndex i = 0; i < channels.size(); ++i)
        {
        auto boolList = grid.isChannel(channels[i][0], ecut);
        cout << setw(3) << boolList[0] << boolList[1] << boolList[2] << " ";
        }
    cout << endl;


    return 0;
    }
catch (GridayException& e)
    {
    std::cerr << e.what() << std::endl;
    }
catch (std::exception& e)
    {
    std::cerr << e.what() << std::endl;
    }
