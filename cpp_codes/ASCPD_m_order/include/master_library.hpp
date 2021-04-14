#ifndef MASTER_LIBRARY_HPP
#define MASTER_LIBRARY_HPP


#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include <limits>
#include <ctime>
#include <chrono>
#include <random>
#include <bits/stdc++.h> 
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/CXX11/Tensor>


//#define EIGEN_DONT_PARALLELIZE

using namespace Eigen;
using namespace std;
using namespace std::chrono;

#include "omp_lib.hpp"
#include "calc_gradient.hpp"
#include "sampling_funs.hpp"
#include "solve_BrasCPaccel.hpp"
#include "cpdgen.hpp"
#include "khatri_rao_prod.hpp"
#include "sampling_funs.hpp"
#include "mttkrp.hpp"
#include "compute_fval.hpp"


#endif  
