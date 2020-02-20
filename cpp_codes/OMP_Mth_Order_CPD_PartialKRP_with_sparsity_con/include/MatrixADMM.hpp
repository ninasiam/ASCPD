#ifndef PARTENSOR_MATRIX_ADMM_HPP
#define PARTENSOR_MATRIX_ADMM_HPP

#include "master_lib.hpp"
#include "Eigen/Eigenvalues"


void Matrix_ADMM(const Ref<const MatrixXd> X,  Ref<MatrixXd> S, Ref<MatrixXd > W, Ref<VectorXi> matrix_dims, const double lambda, const double rho, double &error)
{
    const double tol = 1e-3;     // | Tolerance for ADMM Algorithm
    const int MAX_INNER_ITER = 1000;
    const int MAX_OUTER_ITER = 100;

    int outer_iter   = 1;
    int inner_iter   = 1;
    double thr_var   = lambda/rho;

    //matrix_dims[0] = N, matrix_dims[1] = M, matrix_dims[2] = R
    
    MatrixXd T(X.rows(), S.cols());
    MatrixXd Y(matrix_dims(0), matrix_dims(2));
    MatrixXd Z(T.rows(), T.cols());
    MatrixXd T_trans_T(matrix_dims(0), matrix_dims(0));
    MatrixXd M_trans_M(matrix_dims(0), matrix_dims(0));
    MatrixXd Identity_matrix = MatrixXd::Identity(S.cols(), S.cols()); 
    VectorXd Ones_vec = VectorXd::Ones(S.cols());
    MatrixXd Matrix_thr_var(Z.rows(), Z.cols());
    Matrix_thr_var.setConstant(thr_var);

    while(1)
    {
        inner_iter   = 1;
        outer_iter++; 
        T = X*S;
        Y.setZero();
        T_trans_T = T.transpose()*T;

        EigenSolver<MatrixXd> eig(T_trans_T); 
        MatrixXd eig_vectors = (eig.eigenvectors()).real();
        VectorXd eig_values = (eig.eigenvalues()).real();

        VectorXd vector  = Ones_vec.cwiseQuotient(eig_values.cwiseSqrt());
        MatrixXd Lambda  = vector.asDiagonal();
        
        Z = T*eig_vectors*Lambda*(eig_vectors.transpose());


        while (1)
        {
            inner_iter++;
            //Update W
            MatrixXd M = 2*T - rho*(Y - Z);
            M_trans_M = M.transpose()*M;

            EigenSolver<MatrixXd> eig(M_trans_M);
            eig_vectors = (eig.eigenvectors()).real();
            eig_values = (eig.eigenvalues()).real();

            vector  = Ones_vec.cwiseQuotient(eig_values.cwiseSqrt());
            Lambda  = vector.asDiagonal();

            W = M*eig_vectors*Lambda*(eig_vectors.transpose());

            //Update Z
            Z = W + Y;

            MatrixXd Z_minus = Z - Matrix_thr_var;
            MatrixXd Z_plus = Z + Matrix_thr_var;

            Z = Z_minus.cwiseMax(MatrixXd::Zero(Z.rows(),Z.cols())) + Z_plus.cwiseMin(MatrixXd::Zero(Z.rows(),Z.cols()));

            //Update Y
            Y = Y + W - Z;

            error = (W - Z).norm();

            if(error < tol || inner_iter > MAX_INNER_ITER )
            {   
                VectorXd tmp = Z.cwiseAbs2().colwise().sum();
                double nnz = 0;
                for (int col_i = 0; col_i < tmp.size(); col_i++)
                {
                    if (tmp(col_i) > 0)
                    {
                        nnz++;
                    }
                } 
                if (nnz == matrix_dims(2))
                {
                    W = Z;
                }
                std::cout << "Break!"<< std::endl;
                break;
            }
            std::cout << "Inner iteration ended!"  << "Error:" << error << std::endl;
            
        }

        S = X.transpose() * W; 

        if (outer_iter > MAX_OUTER_ITER)
        {
            break;
        }
    }
}
#endif
