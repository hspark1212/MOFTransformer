CXX = g++
CXXFLAGS = -O4 -std=c++14 -W -Wall
LIBFLGAS = -L .. -l griday

all: grid_gen grid2visit grid2props grid2COTA

grid_gen: main_make_egrid.cpp
	$(CXX) $(CXXFLAGS) main_make_egrid.cpp $(LIBFLGAS) -o $(@)

grid2visit: main_visit.cpp
	$(CXX) $(CXXFLAGS) main_visit.cpp $(LIBFLGAS) -o $(@)

grid2props: adsorption_properties.cpp
	$(CXX) $(CXXFLAGS) adsorption_properties.cpp $(LIBFLGAS) -o $(@)

grid2COTA: main_cota.cpp
	$(CXX) $(CXXFLAGS) $(EIGENFLAGS) main_cota.cpp $(LIBFLGAS) -o $(@)

clean:
	rm grid_gen && rm grid2visit && rm grid2props && rm grid2COTA
