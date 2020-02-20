#ifndef UPDATEFACTOR_HPP
#define UPDATEFACTOR_HPP

#include "master_lib.hpp"
#include "nesterovNNLS.hpp"
#include "omp_lib.hpp"

/* 
 * void updateFactor(<IN>  const Ref<const MatrixXd> MTTKRP, 
 *                   <IN>  const Ref<const MatrixXd> V, 
 *                   <OUT> MatrixXd                  &Factor,             
 *                   <IN>  const Ref<const VectorXi> constraints, 
 *                   <IN>  const int                 current_mode)
 * 
 * Description: Updates each factor with respect to the given constraints.
 * 
 * param MTTKRP       : contains the MTTKRP of last Factor A_{n},
 * param V            : contains the Hadamard product of all Factors A_{1}... A_{n-1},
 * param Factor       : is the last Factor A_{n},
 * param frob_X       : is the frobenious norm of input Tensor X,
 * param constraints  : contains each factor's constraints,
 * param current_mode : is the current Factor mode in {1, 2, ..., TNS_ORDER}.
 */
void updateFactor(const Ref<const MatrixXd> MTTKRP, const Ref<const MatrixXd> V, MatrixXd &Factor, const Ref<const VectorXi> constraints, const int current_mode)
{
    // Unconstraint. Use closed form solution of Matrix LS.
    if (constraints(current_mode) == 0)
    {
        #ifdef FACTORS_ARE_TRANSPOSED
        MatrixXd Factor_T = MTTKRP * V.inverse();
        Factor = Factor_T.transpose();
        #else
        Factor = MTTKRP * V.inverse();
        #endif
    }
    // Non Negative -- Apply NesterovNNLS Method.
    else if (constraints(current_mode) == 1)
    {
        MatrixXd prevFactor = Factor;
        nesterovNNLS(MTTKRP, V, Factor);
        MatrixXd tmp = Factor.cwiseAbs().colwise().sum();
        // If a column becomes only zeros the algorithm gets unstable. Hence the estimate is weighted with the prior estimate. 
        // This should circumvent numerical problems during the iterations.
        if (tmp.minCoeff() == 0) 
        {
            Factor = 0.9 * Factor + 0.1 * prevFactor;
        }
    }
    // Orthogonality constraints...
    else if (constraints(current_mode) == 2)
    {
        #ifdef FACTORS_ARE_TRANSPOSED
        MatrixXd Hessian(Factor.rows(), Factor.rows());
        Hessian.noalias() = MTTKRP.transpose() * MTTKRP;
        Factor = (MTTKRP * (Hessian.pow(-0.5))).transpose();        
        #else
        MatrixXd Hessian(Factor.cols(), Factor.cols());
        Hessian.noalias() = MTTKRP.transpose() * MTTKRP;
        Factor = MTTKRP * (Hessian.pow(-0.5));        
        #endif
    }
    else
    {
        std::cerr << "Unknown type of constraint for Factor of mode = " << current_mode << std::endl;
    }  
}

#endif