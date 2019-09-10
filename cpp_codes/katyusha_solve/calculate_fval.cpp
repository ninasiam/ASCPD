#include "kat_library.h"
#include <iostream>
using namespace Eigen;
using namespace std;

void Calculate_fval(const MatrixXd &A, const VectorXd &b, const VectorXd &x_init, double *fval){
    
    VectorXd lin_eq;
    int num;

    num = A.rows();

    lin_eq = A*x_init - b;

    // cout << lin_eq.squaredNorm() << endl;

    *fval = (double)lin_eq.squaredNorm()/(2*num);
}