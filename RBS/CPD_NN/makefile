CC = g++
MPICC = $(CC)
OMP = -fopenmp -lnuma
#DEBUG = -g
CXXFLAGS = -march=native -funroll-loops -c -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
# CXXFLAGS = -march=native -c -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
OBJS = khatri_rao_product.o ntf_eigen.o get_objective_value.o get_objective_value_accel.o line_search_accel.o nesterov_matrix_nnls.o normalize.o normalize_init.o read_data.o read_from_file.o set_info.o workerjob.o workerjob_C.o sampling_operator.o
DEPS = ntf_functions.h

all: ntf

ntf: $(OBJS)
	$(MPICC) $(OBJS) -o ntf $(OMP)

ntf_eigen.o: ntf_eigen.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) ntf_eigen.cpp $(OMP)

get_objective_value.o: get_objective_value.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) get_objective_value.cpp $(OMP)

get_objective_value_accel.o: get_objective_value_accel.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) get_objective_value_accel.cpp $(OMP)

khatri_rao_product.o: khatri_rao_product.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) khatri_rao_product.cpp $(OMP)

line_search_accel.o: line_search_accel.cpp get_objective_value_accel.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) line_search_accel.cpp $(OMP)

nesterov_matrix_nnls.o: nesterov_matrix_nnls.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) nesterov_matrix_nnls.cpp $(OMP)

normalize.o: normalize.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) normalize.cpp $(OMP)

normalize_init.o: normalize_init.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) normalize_init.cpp $(OMP)

read_data.o: read_data.cpp read_from_file.cpp read_x_c.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_data.cpp $(OMP)

read_from_file.o: read_from_file.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_from_file.cpp $(OMP)

##read_x_c.o: read_x_c.cpp $(DEPS) $(MPICC) $(CXXFLAGS) read_x_c.cpp

sampling_operator.o: sampling_operator.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) sampling_operator.cpp $(OMP)

set_info.o: set_info.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) set_info.cpp

workerjob.o: workerjob.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) workerjob.cpp $(OMP)

workerjob_C.o: workerjob_C.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) workerjob_C.cpp $(OMP)

create_data: create_data.cpp
	$(CC) -O3 create_data.cpp -o create_data

clean:
	rm *.o *~ ntf create_data

clean_data:
	rm create_data
