#ifndef PARTENSOR_MATRIX_FISTA_HPP
#define PARTENSOR_MATRIX_FISTA_HPP

#include "master_lib.hpp"
#include "Eigen/Eigenvalues"

namespace v1
{
    inline void Soft_Thresholding( Ref<MatrixXd> new_Y, const Ref<const MatrixXd> tmp_A, const double sth_par)
    {
        MatrixXd Matrix_thr_var(tmp_A.rows(), tmp_A.cols());
        Matrix_thr_var.setConstant(sth_par);
        MatrixXd Z_minus = tmp_A - Matrix_thr_var;
        MatrixXd Z_plus = tmp_A + Matrix_thr_var;

        new_Y = Z_minus.cwiseMax(MatrixXd::Zero(tmp_A.rows(),tmp_A.cols())) + Z_plus.cwiseMin(MatrixXd::Zero(tmp_A.rows(),tmp_A.cols()));
        // std::cout << "new_Y = " << new_Y << std::endl;
    }
}

namespace v2
{
    inline void Soft_Thresholding(Ref<MatrixXd> new_Y, const Ref<const MatrixXd> tmp_A, const double sth_par)
    {
        int rows, cols;
        rows = tmp_A.rows();
        cols = tmp_A.cols();

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                
                if(tmp_A(i,j) <= sth_par && tmp_A(i,j) >= -sth_par)
                {
                    
                    new_Y(i,j) = 0.0; 
                }
                else if(tmp_A(i,j) > sth_par)
                {
                    new_Y(i,j) = tmp_A(i,j) - sth_par;
                }
                else
                {
                    new_Y(i,j) = tmp_A(i,j) + sth_par;
                }
            }
            
        }
    }
}
inline void Compute_SVD(double *L, double *mu, const Ref<const MatrixXd> Z)
{
    JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
    *L = svd.singularValues().maxCoeff();
    *mu = svd.singularValues().minCoeff();
}

void MatrixFISTA(const Ref<const MatrixXd> MTTKRP, const Ref<const MatrixXd> V, MatrixXd &Factor, const double lambda)
{   
    // V = had(kr(A^(N)...A^(i+1)A^(i-1)...A^(1))^T,kr(A^(N)...A^(i+1)A^(i-1)...A^(1)))
    const double tol = 1e-3;
    const int MAX_ITER = 1000;

    #ifdef FACTORS_ARE_TRANSPOSED
        int m = Factor.cols();
        int r = Factor.rows();
    #else
        int m = Factor.rows();
        int r = Factor.cols();
    #endif

    double L, mu, gamma, eta, tau_next;
    double tau = 0;
    int iter = 0;
    MatrixXd V_init = V;
    MatrixXd MTTKRP_init = -MTTKRP;
    MatrixXd Y = Factor;
    MatrixXd grad(r,m);
    MatrixXd new_A;
    MatrixXd tmp_A;
    MatrixXd new_Y = MatrixXd::Zero(r,m);
    MatrixXd A = Factor;

    Compute_SVD(&L, &mu, V_init);
    

    while(1)
    {
        grad = MTTKRP_init;             // |
        grad.noalias() += A * V_init;   // | grad = W + A * Z.transpose();

        tau_next = 1 + (sqrt(1 + 4*pow(tau,2.0)))/2;
        tmp_A = A - grad / L;
        gamma = (1 - tau) / tau_next;
        v1::Soft_Thresholding(new_Y, tmp_A, lambda / L);
        new_A = (1-gamma)*new_Y + gamma*Y;

        if((new_A - A).norm()  < tol || iter > MAX_ITER)
        {   
            std::cout<<"Termination Condition attained at iter "<< iter <<std::endl;
            break;
        }

        tau = tau_next;
        A = new_A;
        Y = new_Y;
        iter ++;
    }
    Factor = new_A;
}
#endif