A parallel (MPI) implementation of the Nonnegative Tensor Factorization algorithm presented in [1] and [2].
The program is written in c++ and the matrix operations are implemented using the Eigen library.

[1] A. P. Liavas, G. Kostoulas, G. Lourakis, K. Huang, and N.D. Sidiropoulos, “Nesterov-based parallel algorithm for large-scale nonnegative tensor factorization,” 
    in 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans, USA, March 5-9, 2017. IEEE, 2017.
[2] A. P. Liavas, G. Kostoulas, G. Lourakis, K. Huang, and N.D. Sidiropoulos, “Nesterov-based alternating optimization for nonnegative tensor factorization:Algorithm and parallel implementations,”
    IEEE Transactions on Signal Processing, February 2018.
----------------------------------------------------------------------------------------------------------
December 2017
----------------------------------------------------------------------------------------------------------
Authors:
	Athanasios P. Liavas, 
	Georgios Lourakis, 
	Georgios Kostoulas
----------------------------------------------------------------------------------------------------------
Requirement:
	- c++ compiler (tested in gcc/g++)
	- Open MPI
	- Eigen library
----------------------------------------------------------------------------------------------------------
MAKEFILE commands:
	- make			(compiles the program)
	- make clean		(deletes all the executable files in the folder)
	- make create_data	(compiles create_data.cpp)
	- make clean_data	(deletes the executable create_data) 
----------------------------------------------------------------------------------------------------------

1) Create the data of the problem

	We set the size of the tensor (I, J, K) and the rank of the factorization (R) in the file create_data.cpp.
	We create the true latent factors and some initial factors for the algorithm and store them in binary files in the Data_cpp folder.

	compile : make create_data 
	execute : ./create_data

	(Every time we change I, J, K  or R we must delete the executable create_data manually or with make clean_data, and repeat compile, execute)

2) Compile the program:
	make

3) Run the program:
	mpirun -np #cores ntf

