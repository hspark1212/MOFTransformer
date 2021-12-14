#include <string>
#include <iostream>
#include <iomanip>
#include <exception>

#include "../Griday.hpp"
#include "../Vector.hpp"
#include "../Random.hpp"


int
main(int argc, char* argv[])
try {
    using namespace std;

    if (argc < 3)
        {
        cerr << "./convert_cota GRID output"
            << endl;
        return 1;
        }

    string grid_path    {argv[1]};
    string out_path     {argv[2]};

    EnergyGrid grid;
    grid.read(grid_path);
    grid.transformToPartialMap(5000.0);

    //grid.poreBlocking(298.0 * 15.0);

    auto A = grid.getCellLengths()[0];
    auto B = grid.getCellLengths()[1];
    auto C = grid.getCellLengths()[2];

    GIndex a = static_cast<GIndex>(24.0 / A) + 1;
    GIndex b = static_cast<GIndex>(24.0 / B) + 1;
    GIndex c = static_cast<GIndex>(24.0 / C) + 1;

    grid.writeCotaGrid(out_path, 0.15, 12.0, a, b, c);


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
