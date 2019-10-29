#include <iostream>
#include "kat_library.h"

using namespace Eigen;
using namespace std;

VectorXd Compute_gradient(const MatrixXd &A_t, const MatrixXd &A_t_A, const VectorXd &b, const VectorXd &x, int d, int n){

     VectorXd grad;

     grad = (A_t_A*x - A_t*b)/n;

     // cout << "Full Gradient:" << grad << endl;
     
     return grad;

}