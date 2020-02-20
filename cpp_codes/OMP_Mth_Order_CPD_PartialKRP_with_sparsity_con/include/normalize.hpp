#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include "master_lib.hpp"

/* 
 * void normalize(<IN>  const Ref<const VectorXi>       tensor_dims, 
 *                <IN>  const int                       tensor_rank,        
 *                <OUT> std::array<MatrixXd, TNS_ORDER> &Factors)
 * 
 * Description: Function for the Normalization of the columns of the Factors {2, ... , N}.
 *              All the "weight" is concentrated in the columns of Factor 1.
 */
template <std::size_t TNS_SIZE>
void normalize(const Ref<const VectorXi> tensor_dims, const int tensor_rank, std::array<MatrixXd, TNS_SIZE> &Factors)
{
    double   f_i;
    VectorXd f_i_v(tensor_rank);
    MatrixXd lambdas_fi(tensor_rank, TNS_SIZE);
    MatrixXd cov_factor(tensor_rank, tensor_rank);
    // MatrixXd norm_cov_factor(tensor_rank, tensor_rank);
    // MatrixXd norm_product = MatrixXd::Ones(tensor_rank, tensor_rank);

    // Begin with Factor_(2) ... Factor_(N)
    for (int dim_i = 1; dim_i < TNS_SIZE; dim_i++)
    {
        #ifdef FACTORS_ARE_TRANSPOSED
            cov_factor            = Factors[dim_i] * Factors[dim_i].transpose();
        #else
            cov_factor            = Factors[dim_i].transpose() * Factors[dim_i];
        #endif
        lambdas_fi.col(dim_i) = cov_factor.diagonal();
        f_i_v = lambdas_fi.col(dim_i); // Each column corresponds to each factor.

        for(int rank_r=0; rank_r<tensor_rank; rank_r++)
        {
            f_i = sqrt(f_i_v(rank_r));
            if(f_i > 0)
            {
                // Normalization of column "rank_r" of current factor.
                #ifdef FACTORS_ARE_TRANSPOSED
                    Factors[dim_i].row(rank_r) *= (1/f_i);
                    Factors[0].row(rank_r)     *= f_i;
                #else 
                    Factors[dim_i].col(rank_r) *= (1/f_i);
                    Factors[0].col(rank_r)     *= f_i;
                #endif
                lambdas_fi(rank_r, dim_i)   = f_i;
            }
            else
            {
                lambdas_fi(rank_r, dim_i)   = 1;
            }
        }
        
        // norm_cov_factor = lambdas_fi.col(dim_i) * lambdas_fi.col(dim_i).transpose();
        // cov_factor = cov_factor.cwiseQuotient(norm_cov_factor);
        // norm_product.noalias() = norm_product.cwiseProduct(norm_cov_factor);
    }


}


#endif