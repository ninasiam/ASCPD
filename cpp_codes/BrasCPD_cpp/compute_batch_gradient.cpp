#include <iostream>
#include <random>
#include <cstdlib>
#include "brascpd_functions.h"

using namespace std;
using namespace Eigen;

void Calculate_Batch_Gradient(int batch, const MatrixXd &U_prev, const MatrixXd &KhatriRao_sub, const MatrixXd &X_sub, MatrixXd &Gradient)
{   
    int J_n = KhatriRao_sub.cols();
    int R = KhatriRao_sub.rows();

    MatrixXd H(J_n,R);
    H = KhatriRao_sub.transpose();                          //actual khatri rao H (we calculated H_T)
    
    Gradient = (1/batch)*(U_prev*KhatriRao_sub*H - X_sub.transpose()*H);
}