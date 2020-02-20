#ifndef FTKRP_HPP
#define FTKRP_HPP

#include "master_lib.hpp"
#include "omp_lib.hpp"

/* 
 * void ftkrp(<IN>  const Ref<const MatrixXd> Factor, 
 *            <IN>  const Ref<const MatrixXd> KhatriRao, 
 *            <OUT> MatrixXd                  &Tensor_X,           
 *            <IN>  const Ref<const VectorXi> tensor_dims, 
 *            <IN>  const int                 current_mode)
 * 
 * Description: Computes Factor Times Khatri-Rao Product (Matricized Tensor).
 * 
 * param Factor       : is the current input Factor A_{current_mode},
 * param KhatriRao    : is the Khatri-Rao Product for Factor A_{current_mode},
 * param Tensor_X     : is the output matricized Tensor X,
 * param tensor_dims  : is the vector containing the tensor's dimensions,
 * param current_mode : is the current Factor mode in {1, 2, ..., TNS_ORDER}.

 */

void ftkrp(const Ref<const MatrixXd> Factor, const Ref<const MatrixXd> KhatriRao,
           MatrixXd &Tensor_X, const Ref<const VectorXi> tensor_dims,
           const int current_mode, const unsigned int num_threads)
{
    #ifndef EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(1);
    #endif
    VectorXi reduced_tensor_dim = tensor_dims;
    reduced_tensor_dim(current_mode) = 1;
    int max_dim = reduced_tensor_dim.maxCoeff();

    int offset = reduced_tensor_dim.prod();
    offset = (offset / max_dim);

    #ifdef FACTORS_ARE_TRANSPOSED
        MatrixXd Factor_T = Factor.transpose();
        #pragma omp parallel for default(shared) num_threads(num_threads)
        for (int block = 0; block < max_dim; block++)
        {
            Tensor_X.block(0, block * offset, Factor.cols(), offset) = Factor_T * KhatriRao.block(0, block * offset, KhatriRao.rows(), offset);
        }
    #else
        #pragma omp parallel for default(shared) num_threads(num_threads)
        for (int block = 0; block < max_dim; block++)
        {
            Tensor_X.block(0, block * offset, Factor.rows(), offset) = Factor * KhatriRao.block(0, block * offset, KhatriRao.rows(), offset);
        }
    #endif


    #ifndef EIGEN_DONT_PARALLELIZE
        Eigen::setNbThreads(num_threads);
    #endif 
}

#endif