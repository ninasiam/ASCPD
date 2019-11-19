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
    // cout<< Gradient.rows() << " " << Gradient.cols() << endl;
    Gradient = (U_prev*KhatriRao_sub*H - X_sub*H)/batch;
    // cout << Gradient << endl;
    // cout<< Gradient.rows() << " " << Gradient.cols() << " = "  << "[" << U_prev.rows() << " "  << U_prev.cols() << " -- \t --" << KhatriRao_sub.rows() 
    //      << " " << KhatriRao_sub.cols() << " -- \t --" << H.rows() << " "  << H.cols() << "] -- \t --" << " [" 
    //      << X_sub.cols() << " "  << X_sub.rows() << " -- \t --" << H.rows() << " "  << H.cols() << "]" << endl;
    // Gradient = MatrixXd::Random(10,R);
    // Gradient = Gradient.cwiseAbs();
    // cout << "MPIKE" << endl;
}