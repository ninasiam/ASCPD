#include <iostream>
#include <random>
#include <cstdlib>
#include "brascpd_functions.h"

using namespace std;
using namespace Eigen;

void Sampling_Sub_Matrices(const VectorXi &F_n, MatrixXd &KhatriRao, const MatrixXd &X, const MatrixXd &U1, const MatrixXd &U2,  MatrixXd &KhatriRao_sub, MatrixXd &X_sub)
{   
    int R = KhatriRao.cols();
    int J_n = KhatriRao.rows();
    int bz = F_n.size();
    MatrixXd KhatriRao_T(R, J_n);

    Khatri_Rao_Product(U1, U2, KhatriRao);                          // Compute the full Khatri-Rao Product
    cout << KhatriRao << endl;
    KhatriRao_T = KhatriRao.transpose();

    for(int col_H = 0; col_H < bz; col_H++)
    {
        KhatriRao_sub.col(col_H) = KhatriRao_T.col(F_n(col_H));     //Create KhatriRao_sub (transpose)
    }
    
    for(int col_X = 0; col_X < bz; col_X++)
    {
        X_sub.col(col_X) = X.col(F_n(col_X));
    }
}