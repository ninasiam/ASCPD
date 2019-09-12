CC = g++
MPICC = $(CC)
OBJS = ERM_min.o compute_parameters.o calculate_fval.o compute_gradient.o random_int_generator.o
DEPS = kat_library.h

.PHONY = all clean

all: katyusha

katyusha: $(OBJS)
	$(MPICC) $(OBJS) -o katyusha #line begins with a tab!!!
ERM_min.o: ERM_min.cpp $(DEPS)
	$(MPICC) -c ERM_min.cpp
compute_parameters.o: compute_parameters.cpp $(DEPS)
	$(MPICC) -c compute_parameters.cpp
calculate_fval.o: calculate_fval.cpp $(DEPS)
	$(MPICC) -c calculate_fval.cpp 
compute_gradient.o: compute_gradient.cpp $(DEPS)
	$(MPICC) -c compute_gradient.cpp 
random_int_generator.o: random_int_generator.cpp $(DEPS)
	$(MPICC) -c random_int_generator.cpp
clean:
	@echo "Cleaning Up..."
	rm *.o katyusha 