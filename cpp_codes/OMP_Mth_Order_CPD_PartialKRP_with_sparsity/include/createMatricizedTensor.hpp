#ifndef CREATE_MATRICIZED_TENSOR_HPP
#define CREATE_MATRICIZED_TENSOR_HPP

#include "master_lib.hpp"
#include "tensor2matrix.hpp"

/* 
 * void createRandomTensor(<IN>  const int                        tensor_order,                       
 *                         <IN>  const Ref<const VectorXi>        tensor_dims, 
 *                         <IN>  Eigen::Tensor<double, TNS_ORDER> &True_Tensor, 
 *                         <OUT> std::array<MatrixXd, TNS_ORDER>  &True_Tensor_Mat)
 * 
 * Description: Generates Matricized Tensors from input Tensor.
 * 
 * param tensor_order    : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param True_Tensor     : is the input Tensor X of type Tensor,
 * param True_Tensor_Mat : is the matricized output Tensor of type MatrixXd.
 */

template <int TNS_SIZE, std::size_t TNS_ORDER>
void createMatricizedTensor(const int tensor_order, const Ref<const VectorXi> tensor_dims,
                            Eigen::Tensor<double, TNS_SIZE> True_Tensor, std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat)
{
    /*-- Start with X_(2) ... --*/
    for(int dim_i=1; dim_i<tensor_order; dim_i++)
    {
        // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
        int dim = tensor_dims.prod() / tensor_dims(dim_i);

        True_Tensor_Mat[dim_i] = MatrixXd::Zero(tensor_dims(dim_i), dim);

        tensor2matrix(True_Tensor, True_Tensor_Mat[dim_i], tensor_dims, dim_i);
        // std::cout << dim_i << "\n" << True_Tensor_Mat[dim_i] << std::endl;
    }
}

#endif