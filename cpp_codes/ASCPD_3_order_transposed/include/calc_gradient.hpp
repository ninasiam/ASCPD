#ifndef CALC_GRADIENT_HPP
#define CALC_GRADIENT_HPP

#include "master_library.hpp"
#include "mttkrp.hpp"

inline void Compute_SVD(double &L, double &mu, const MatrixXd &Z)
{
    JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
    L = svd.singularValues().maxCoeff();
    mu = svd.singularValues().minCoeff();
}

inline void Compute_NAG_parameters(const MatrixXd &Hessian, double &L, double  &beta, double &lambda)
{   
    double mu, cond, Q;
    Compute_SVD(L, mu, Hessian);

    cond = L/(mu + 1e-6);

    if(cond> 1e5)
    {
        lambda = L/1000;
    }
    else
    {
        lambda = 0;
    }

    Q = (mu + lambda)/(L + lambda);
    beta = (1 - sqrt(Q))/(1 + sqrt(Q));
}

inline void Calc_gradient(const VectorXi &Tns_dims, int Mode, const unsigned int thrds,
                          const double lambda,  const MatrixXd &U_prev, const MatrixXd &Y, 
                          const MatrixXd &Hessian, const MatrixXd &H, const MatrixXd &X_sub, MatrixXd &Gradient)
{   

    int R = Hessian.rows();
    int rows_mttkrp = X_sub.rows();

    MatrixXd MTTKRP(rows_mttkrp,R);                            // I_n * R


    mttkrp( X_sub, H, Tns_dims, Mode, thrds, MTTKRP);

    Gradient = Y*(Hessian + lambda*(MatrixXd::Identity(R,R)))-(MTTKRP + lambda*U_prev);

}
#endif