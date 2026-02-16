CXX = g++
CXXFLAGS = -O3 -fopenmp -mavx2 -mfma -Wall -std=c++11
TARGET = correlate_app
SRCS = main.cpp correlate.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
