CC = g++
GCC = $(CC)
OMP = -fopenmp #-lnuma
#DEBUG = -g
CXXFLAGS = -march=native -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
OBJS = ../include/*.hpp

all: brasCP_NN_test

brasCP_NN_test: brasCPD_NonNegative.cpp $(OBJS)
	$(GCC) $(CXXFLAGS) brasCPD_NonNegative.cpp -o brasCP_NN_test $(OMP)

clean:
	@echo "Cleaning Up..."
	rm *.o brasCP_NN_test  

clean_data:
	rm create_data
