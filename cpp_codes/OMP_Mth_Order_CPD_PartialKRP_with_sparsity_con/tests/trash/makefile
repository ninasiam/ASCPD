CC = g++ -std=c++11 -w
GCC = $(CC)
OMP = -fopenmp #-lnuma
#DEBUG = -g
CXXFLAGS = -mtune=native -march=native -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
DEPS = ../include/*.hpp

all: test1.o test2.o test_create_data.o

test1.o: test_run_alscpd_from_file.cpp $(DEPS)
	$(GCC) $(CXXFLAGS) test_run_alscpd_from_file.cpp -o test1.o $(OMP)

test2.o: test_writeread_matricization_tofrom_file.cpp $(DEPS)
	$(GCC) $(CXXFLAGS) test_writeread_matricization_tofrom_file.cpp -o test2.o

test_create_data.o: test_create_tensor_from_factors.cpp $(DEPS)
	$(GCC) $(CXXFLAGS) test_create_tensor_from_factors.cpp -o test_create_data.o $(OMP)

test3.o: test_run_alscpd_with_random_factors_at_runtime.cpp $(DEPS)
	$(GCC) $(CXXFLAGS) test_run_alscpd_with_random_factors_at_runtime.cpp -o test3.o $(OMP)

clean:
	rm *.o 

clean_dat:
	rm *.dat