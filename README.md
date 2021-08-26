# ASCPD and stochastic algorithms for Canonical Polyadic Decomposition (CPD)

## About the project:
The Accelerated Stochastic Canonical Polyadic Decomposition algorithm (ASCPD) is an accelerated algorithm for Canonical Tensor Decomposition Problem. It combines ALS with a fiber sampling technique in order to reduce the computational cost of the operation Matricized Tensor Times Khatri Rao Product (MTTKRP). \
The project contains:
   - matlab_codes directory, with scripts for testing and prototyping stochastic algorithms for tensor factorization problems.
   - cpp_codes directory, with two versions of ASCPD (3-order, M-order).
     - Tests are availables to test the functionality of the implemented functions.

## Built with:
For the matlab scripts the following libraries were used:
  - tensorlab: a matlab library for tensor operations.

For the C++ project:
  - Eigen library: A C++ library, for linear algebra operations.
  - Tensor module (Eigen Library): a module that provides a Tensor data structure. Note that the Tensor module is not supported.
##
###### The ASCPD algorithm was presented  on the paper "AN ACCELERATED STOCHASTIC GRADIENT FOR CANONICAL POLYADIC DECOMPOSITION" on Eusipco 2021.
