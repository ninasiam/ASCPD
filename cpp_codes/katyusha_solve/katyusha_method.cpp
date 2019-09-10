#include <iostream>
#include "kat_library.h"

using namespace Eigen;
using namespace std;

void Katyusha_method(const MatrixXd &A, const VectorXd &b, const VectorXd &x_init, int S, double sigma, double L, int n, int d, VectorXd *x_star){
    
    int m;
    float tau_2;
    double tau_1, alpha;
    VectorXd y, z, x_tilda;
    VectorXd mu;
    MatrixXd A_t(d,n)

    m = 2*A.rows();
    tau_2 = 0.5;
    tau_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
    alpha = 1/(3*tau_1*L);

    y = x_init;
    z = x_init;
    x_tilda = x_init;

    for(int s = 0; s < S; s++){

        mu(s) = compute_gradient();

    }

}