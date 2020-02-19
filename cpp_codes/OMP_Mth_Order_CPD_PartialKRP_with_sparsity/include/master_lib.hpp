#ifndef MASTER_LIB_HPP // Guard: if master_lib.hpp hasn't been included yet...
#define MASTER_LIB_HPP // #define this so the compiler knows it has been included

#include <iomanip>
#include <fstream>
#include <iostream>
#include <time.h>
#include <math.h>
#include <string>
#include <limits>

#include <array>

#define EIGEN_DONT_PARALLELIZE
#define PRINT_INFO
#define FACTORS_ARE_TRANSPOSED

/* Uncomment if running at DALI*/
// #include "/usr/local/include/eigen3/Eigen/Dense"
// #include "/usr/local/include/eigen3/Eigen/Core"
// #include "/usr/local/include/eigen3/unsupported/Eigen/MatrixFunctions"
// #include "/usr/local/include/eigen3/unsupported/Eigen/CXX11/Tensor"
/* Uncomment if Eigen is allready installed...*/
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/CXX11/Tensor>
/* Uncomment if running at ARIS-HPC*/
// #include "../../eigen3/Eigen/Dense"
// #include "../../eigen3/Eigen/Core"
// #include "../../eigen3/unsupported/Eigen/MatrixFunctions"
// #include "../../eigen3/unsupported/Eigen/CXX11/Tensor"

// /* -- Tensor Order -- */ 
// #define TNS_ORDER 3

using namespace Eigen;

#endif
