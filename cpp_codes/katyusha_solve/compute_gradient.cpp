#include <iostream>
#include "kat_library.h"

using namespace Eigen;
using namespace std;

VectorXd Compute_gradient(const MatrixXd &A_t, const MatrixXd &A_t_A, const VectorXd &b, const VectorXd &x, int d, int n){

     VectorXd grad;

     grad = (1/2*n)*(A_t_A*x - A_t*b);

     cout << "Fulle Gradient:" << grad << endl;
     
     return grad;

}