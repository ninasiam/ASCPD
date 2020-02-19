#ifndef NESTEROVNNLS_HPP
#define NESTEROVNNLS_HPP

#include "master_lib.hpp"
#include "omp_lib.hpp"

inline void update_alpha(double alpha, double q, double *new_alpha)
{
    double a, b, c, D;
    a = 1;
    b = alpha * alpha - q;
    c = -alpha * alpha;
    D = b * b - 4 * a * c;

    *new_alpha = (-b + sqrt(D)) / 2;
}

inline void G_Lambda(double *lambda, double *q, double *L, double mu)
{
    *q = mu / (*L);
    if (1 / (*q) > 1e6)
        *lambda = 10 * mu;
    else if (1 / (*q) > 1e3)
        *lambda = mu;
    else
        *lambda = mu / 10;

    *L += (*lambda);
    mu += (*lambda);
    *q = mu / (*L);
}

inline void Compute_SVD(double *L, double *mu, const Ref<const MatrixXd> Z)
{
    JacobiSVD<MatrixXd> svd(Z, ComputeThinU | ComputeThinV);
    *L = svd.singularValues().maxCoeff();
    *mu = svd.singularValues().minCoeff();
}

void nesterovNNLS(const Ref<const MatrixXd> MTTKRP, const Ref<const MatrixXd> V, MatrixXd &Factor)
{
    const double delta_1 = 1e-2; // | Tolerance for Nesterov Algorithm
    const double delta_2 = 1e-2; // |
    #ifdef FACTORS_ARE_TRANSPOSED
        int m = Factor.cols();
        int r = Factor.rows();
    #else
        int m = Factor.rows();
        int r = Factor.cols();
    #endif
    double L, mu, lambda, q, alpha, new_alpha, beta;

    MatrixXd grad_Y(m, r);
    MatrixXd Y(m, r);
    MatrixXd new_A(m, r);
    MatrixXd A(m, r);
    MatrixXd Zero_Matrix = MatrixXd::Zero(m, r);
    MatrixXd MTTKRP_init = - MTTKRP;
    MatrixXd V_init = V;
    Compute_SVD(&L, &mu, V_init);

    G_Lambda(&lambda, &q, &L, mu);

    V_init.noalias() += lambda * MatrixXd::Identity(r, r);
    alpha = 1;
    #ifdef FACTORS_ARE_TRANSPOSED
        MTTKRP_init.noalias() -= lambda * Factor.transpose();
        A = Factor.transpose();
        Y = Factor.transpose();        
    #else
        MTTKRP_init.noalias() -= lambda * Factor;
        A = Factor;
        Y = Factor;        
    #endif

    while (1)
    {
        grad_Y = MTTKRP_init;      // |
        grad_Y.noalias() += Y * V_init; // | grad_Y = W + Y * Z.transpose();

        if (grad_Y.cwiseProduct(Y).cwiseAbs().maxCoeff() <= delta_1 && grad_Y.minCoeff() >= -delta_2)
            break;

        new_A = (Y - grad_Y / L).cwiseMax(Zero_Matrix);

        update_alpha(alpha, q, &new_alpha);
        beta = alpha * (1 - alpha) / (alpha * alpha + new_alpha);

        Y = (1 + beta) * new_A - beta * A;

        A = new_A;
        alpha = new_alpha;
    }
    #ifdef FACTORS_ARE_TRANSPOSED
        Factor = A.transpose();
    #else
        Factor = A;
    #endif
}

#endif