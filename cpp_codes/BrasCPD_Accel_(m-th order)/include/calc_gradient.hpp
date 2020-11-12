#ifndef CALC_GRADIENT_HPP
#define CALC_GRADIENT_HPP

#include "master_library.hpp"
#include "mttkrp.hpp"


namespace svd
{
    inline void Compute_mu_L(double &L, double &mu, const MatrixXd &Z)
    {   
        int R = Z.cols();
        if(R <= 16)
        {
            JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
            L = svd.singularValues().maxCoeff();
            mu = svd.singularValues().minCoeff();
        }
        else
        {
            BDCSVD<MatrixXd> svd(Z);
            L = svd.singularValues().maxCoeff();
            mu = svd.singularValues().minCoeff();
        }    
    }
        
}

namespace eig
{   
    inline void Compute_mu_L(double &L, double &mu, const MatrixXd &Z)
    {
        EigenSolver<MatrixXd> es(Z, false);
        L = (es.eigenvalues().real()).maxCoeff();
        mu = (es.eigenvalues().real()).minCoeff();
    }
}



inline void Compute_NAG_parameters(const MatrixXd &Hessian, double &L, double  &beta, double &lambda)
{   
    double mu, cond, Q;
    eig::Compute_mu_L(L, mu, Hessian);

    cond = L/(mu + 1e-6);
    if(cond> 1e2)
    {
        lambda = L/100;
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
                          const MatrixXd &Hessian, const MatrixXd &H, const MatrixXd &X_sub, MatrixXd &Gradient, nanoseconds &time_MTTKRP)
{   


    int R = Hessian.rows();
    int rows_mttkrp = X_sub.rows();
   
    MatrixXd MTTKRP(rows_mttkrp,R);                            // I_n * R
 
    
    // Begin timer MTTKRP
    auto t1_MTTKRP = high_resolution_clock::now();
    MTTKRP = X_sub*H;
    // v1::mttkrp( X_sub, H, Tns_dims, Mode, thrds, MTTKRP);
    // End timer MTTKRP
    auto t2_MTTKRP = high_resolution_clock::now();
    time_MTTKRP += duration_cast<nanoseconds>(t2_MTTKRP - t1_MTTKRP);

    Gradient = Y*(Hessian + lambda*(MatrixXd::Identity(R,R)))-(MTTKRP + lambda*U_prev);

}
#endif