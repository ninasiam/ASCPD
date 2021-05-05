# Main repo for the development of ASCPD and stochastic algorithms for Canonical Polyadic Decomposition(CPD).

## About the project:
The ASCPD algorithm is an accelerated algorithm for Canonical Tensor Decomposition Problem. It combines ALS with a fiber sampling technique in order to reduce to alleviate the computational cost of the operation demanding Matricized Tensor Times Khatri Rao Product (MTTKRP). 
The project contains:
   - matlab_codes directory, with scripts for testing and prototyping stochastic algorithms for tensor factorization problems.
   - cpp_codes directory, with two versions of ASCPD (3-order, M-order).
     - Tests are availables to test the functionality of the implemented functions.

## Built with:
For the matlab scripts the following libraries were used:
  - tensorlab: a matlab library for tensor operations.

For the C++ project:
  - Eigen library: A C++ library, for linear algebra operations.
  - Tensor module (Eigen Library): a module that provides a Tensor data structure.

