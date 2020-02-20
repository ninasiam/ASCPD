#ifndef CREATE_TENSOR_FROM_FACTORS_HPP
#define CREATE_TENSOR_FROM_FACTORS_HPP

#include "master_lib.hpp"
#include "khatri_rao_product.hpp"
#include "FullKhatriRaoProduct.hpp"
#include "ftkrp.hpp"

/* 
 * void create_tensor_from_factors(<IN>  const int                       tensor_order, 
 *                                 <IN>  const Ref<const VectorXi>       tensor_dims, 
 *                                 <IN>  int                             tensor_rank,        
 *                                 <IN>  std::array<MatrixXd, TNS_ORDER> &True_Factors, 
 *                                 <OUT> std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat)
 * 
 * Description: Generates a matricised tensor of order "tensor_order" and rank "tensor_rank" using synthetic factors.
 * 
 * param tensor_order    : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param True_Factors    : contains all input Factors,
 * param True_Tensor_Mat : is the matricized output Tensor X.
 */
template <std::size_t TNS_SIZE>
void create_tensor_from_factors(const int tensor_order, const Ref<const VectorXi> tensor_dims, int tensor_rank,
                                std::array<MatrixXd, TNS_SIZE> &True_Factors, std::array<MatrixXd, TNS_SIZE> &True_Tensor_Mat,
                                const unsigned int num_threads)
{
    for (int mode = 0; mode < tensor_order; mode++)
    {
        // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
        int dim = tensor_dims.prod() / tensor_dims(mode);

        MatrixXd Khatri_Rao = MatrixXd::Zero(tensor_rank, dim);

        int mode_N = tensor_order - 1;
        if (mode == mode_N)
        {
            mode_N = mode - 1;
        }

        FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, mode, True_Factors, Khatri_Rao, get_num_threads());

        True_Tensor_Mat[mode] = MatrixXd::Zero(tensor_dims(mode), dim);

        ftkrp(True_Factors[mode], Khatri_Rao, True_Tensor_Mat[mode], tensor_dims, mode, num_threads);
    }
}

#endif