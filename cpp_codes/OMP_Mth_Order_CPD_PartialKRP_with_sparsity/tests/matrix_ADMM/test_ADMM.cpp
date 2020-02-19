#include "../../include/master_lib.hpp"
#include "../../include/MatrixADMM.hpp"

#include <string>

/* TEST: writeMode1Matricization(...) and readMode1Matricization(...)  */
int main(int argc, char **argv)
{
    int N = 10000; 
    int M = 100;
    int R = 3;

    MatrixXd X = MatrixXd::Random(N, M);
    MatrixXd S = (N*R)*MatrixXd::Random(M, R);
    MatrixXd W(N, R);

    double lambda = 20;
    double rho = 10000;

    VectorXi matrix_dims(3);
    matrix_dims(0) = N;
    matrix_dims(1) = M;
    matrix_dims(2) = R;

    Matrix_ADMM(X, S, W,matrix_dims, lambda, rho);

}