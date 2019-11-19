CC = g++
MPICC = $(CC)
#DEBUG = -g
CXXFLAGS = -march=native -funroll-loops -c -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
# CXXFLAGS = -march=native -c -O3 -DNDEBUG #-fopenmp # -Wall# $(DEBUG)#LFLAGS = -Wall# $(DEBUG)
OBJS = khatri_rao_product.o brasCPD_NonNegative.o read_data.o read_from_file.o set_info.o get_objective_value.o sampling_operator.o sampling_sub_matrices.o compute_batch_gradient.o
DEPS = brascpd_functions.h

all: brasCP_NN

brasCP_NN: $(OBJS)
	$(MPICC) $(OBJS) -o brasCP_NN 

brasCPD_NonNegative.o: brasCPD_NonNegative.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) brasCPD_NonNegative.cpp 

khatri_rao_product.o: khatri_rao_product.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) khatri_rao_product.cpp 

get_objective_value.o: get_objective_value.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) get_objective_value.cpp

sampling_operator.o: sampling_operator.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) sampling_operator.cpp

read_data.o: read_data.cpp read_from_file.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_data.cpp 

read_from_file.o: read_from_file.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) read_from_file.cpp

set_info.o: set_info.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) set_info.cpp

sampling_sub_matrices.o: sampling_sub_matrices.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) sampling_sub_matrices.cpp

compute_batch_gradient.o: compute_batch_gradient.cpp $(DEPS)
	$(MPICC) $(CXXFLAGS) compute_batch_gradient.cpp

create_data: create_data.cpp
	$(CC) -O3 create_data.cpp -o create_data

clean:
	@echo "Cleaning Up..."
	rm *.o brasCP 

clean_data:
	rm create_data
