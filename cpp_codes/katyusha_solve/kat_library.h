#ifndef KAT_LIBRARY_H	    // if cpd_functions.h hasn't been included yet...
#define KAT_LIBRARY_H		// #define this so the compiler knows it has been included

//#define EIGEN_DEFAULT_TO_ROW_MAJOR

// Run locally
// #include <Eigen/Dense>
// #include <Eigen/Core>

// Run on dali
#include "/home/ninasiam/Desktop/PARTENSOR/Libraries/eigen3/Eigen/Dense"
#include "/home/ninasiam/Desktop/PARTENSOR/Libraries/eigen3/Eigen/Core"
using namespace Eigen;


// void Katyusha_method(const VectorXd &x_init, int S, double Lambda, double sigma);

void Compute_parameters(const MatrixXd &A_t_A, double *L, double *sigma);

void Calculate_fval(const MatrixXd &A, const VectorXd &b, const VectorXd &x_init, double *fval);

VectorXd Compute_gradient(const MatrixXd &A_t, const MatrixXd &A_t_A, const VectorXd &b, const VectorXd &x, int d, int n);

int Random_int_generator(int max);

#endif
