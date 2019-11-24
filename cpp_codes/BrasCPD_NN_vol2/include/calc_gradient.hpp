#ifndef CALC_GRADIENT_HPP
#define CALC_GRADIENT_HPP

#include "master_library.hpp"
#include "mttkrp.hpp"


inline void Calc_gradient(int batch, const VectorXi &Tns_dims, int Mode, const unsigned int thrds,  const MatrixXd &U_prev, const MatrixXd &KhatriRao_sub, const MatrixXd &X_sub, MatrixXd &Gradient)
{   
    int J_n = KhatriRao_sub.cols();
    int R = KhatriRao_sub.rows();
    int rows_mttkrp = X_sub.rows();

    MatrixXd H(J_n,R);
    MatrixXd MTTKRP(rows_mttkrp,J_n);
    H = KhatriRao_sub.transpose();                          //actual khatri rao H (we calculated H_T)
    // cout<< Gradient.rows() << " " << Gradient.cols() << endl;

    mttkrp( X_sub, H, Tns_dims, Mode, thrds, MTTKRP);
    Gradient = (U_prev*KhatriRao_sub*H - MTTKRP)/batch;

}
#endif