#ifndef GENERATE_TENSOR_AT_RUNTIME_HPP
#define GENERATE_TENSOR_AT_RUNTIME_HPP

#include "master_lib.hpp"

/* 
 * generateRandomTensor(<IN>  const Ref<const VectorXi>        tensor_dims, 
 *                      <OUT> Eigen::Tensor<double, TNS_ORDER> &TNS)
 * 
 * Description: Generates a tensor of order "TNS_ORDER" with dimensions "tensor_dims" defined during runtime.
 */
template <int TNS_SIZE>
inline void generateTensorAtRuntime(const Ref<const VectorXi> tensor_dims, Eigen::Tensor<double, TNS_SIZE> &TNS)
{
    // Eigen::Tensor<double, TNS_ORDER> TNS;
    TNS.resize(tensor_dims);
    // TNS.setRandom();
}

#endif