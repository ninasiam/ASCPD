#ifndef CAL_FVAL_HPP
#define CAL_FVAL_HPP

#include <omp.h>
#include "master_library.hpp"
#include "omp_lib.hpp"

// gram_cwise_prod matrix must should be initialised to ones
template <std::size_t TNS_ORDER> 
void compute_gram_cwise_prod(const std::array<MatrixXd, TNS_ORDER> &Factors, MatrixXd &gram_cwise_prod)
{   
    // Compute Hadamard product of Grammian (A{1}_T*A{1} .* .... A{N}_T*A{N})
    MatrixXd factor_T_factor;
    
    for(int factor = 0; factor < TNS_ORDER; factor++)
    {
        factor_T_factor = Factors[factor].transpose()*Factors[factor];
        gram_cwise_prod.noalias() = gram_cwise_prod.cwiseProduct(factor_T_factor);
    }
}

// Compute f_value 
void compute_fval(double frob_X, MatrixXd &MTTKRP, MatrixXd &V, const MatrixXd &Factor, double &f_val)
{   
    
    double sum_MTTKRP = (MTTKRP.cwiseProduct(Factor)).sum();
    double sum_V = V.sum();
    f_val = frob_X + sum_V - 2*sum_MTTKRP;
}
#endif