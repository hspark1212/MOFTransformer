CXX = g++
CXXFLAGS = -O4 -std=c++14 -W -Wall

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

TARGET = libgriday.a

all: $(TARGET)

# Make excutable file
$(TARGET): $(OBJS)
	ar rc $(@) $(OBJS)
#	$(CXX) $(CXXFLAGS) $(^) -o $(@)

# Make .o files from .cpp file
.cpp.o: $(SRCS)
	$(CXX) $(CXXFLAGS) -c $(<)

clean:
	rm $(OBJS) $(TARGET)
