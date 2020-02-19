#ifndef COMPUTE_FVAL_HPP
#define COMPUTE_FVAL_HPP

#include "master_lib.hpp"
#include "omp_lib.hpp"

/* 
 * <OUT> double compute_fval(<IN> const Ref<const MatrixXd> MTTKRP, 
 *                           <IN> const Ref<const MatrixXd> Factor, 
 *                           <IN> const Ref<const MatrixXd> V,      
 *                           <IN> double                    frob_X)
 * 
 * Description: Computes cost function f_X (used as quality measure for CPD).
 * 
 * param MTTKRP : contains the MTTKRP of last Factor A_{n},
 * param Factor : is the last Factor A_{n},
 * param V      : containts the Hadamard product of all Factors A_{1}... A_{n-1},
 * param frob_X : is the frobenious norm of input Tensor X.
 */

inline double compute_fval(const Ref<const MatrixXd> MTTKRP, const Ref<const MatrixXd> Factor, 
                           const Ref<const MatrixXd> V,      const double frob_X)
{
    #ifdef FACTORS_ARE_TRANSPOSED
        /*--+ Compute Gramian Matrix as : Gramian = Hadamard(V, Factor_(N)^T * Factor_(N)) +--*/
        MatrixXd Gramian(Factor.rows(), Factor.rows());
        Gramian.noalias() = Factor*Factor.transpose();
        Gramian = V.cwiseProduct(Gramian);
        double sum = (MTTKRP.cwiseProduct(Factor.transpose())).sum();
        double fval = frob_X + Gramian.sum() - 2*sum;
        return sqrt(fval / frob_X);
    #else
        /*--+ Compute Gramian Matrix as : Gramian = Hadamard(V, Factor_(N)^T * Factor_(N)) +--*/
        MatrixXd Gramian(Factor.cols(), Factor.cols());    
        Gramian = Factor.transpose()*Factor;
        Gramian = V.cwiseProduct(Gramian);
        double sum = (MTTKRP.cwiseProduct(Factor)).sum();
        double fval = frob_X + Gramian.sum() - 2*sum;
        return sqrt(fval / frob_X);
    #endif
}

#endif