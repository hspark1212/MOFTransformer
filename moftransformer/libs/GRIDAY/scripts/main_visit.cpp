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
        cerr << "./visit_maker GRID OUTPUT (blocking)"
            << endl;
        return 1;
        }

    string grid_path    {argv[1]};
    string out_path     {argv[2]};
    string blocking     {};
    if (argc > 3)
        blocking =  {argv[3]};

    EnergyGrid grid;
    grid.read(grid_path);
    //grid.transformToPartialMap(5000.0);

    if (blocking == "blocking")
        {
        grid.poreBlocking(4500.0);
        }

    grid.writeVisitInput(out_path, true);

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
