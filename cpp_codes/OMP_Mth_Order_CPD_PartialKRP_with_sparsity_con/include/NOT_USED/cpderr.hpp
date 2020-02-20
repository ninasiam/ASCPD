#ifndef CPDERR_HPP
#define CPDERR_HPP

#include "master_lib.hpp"
#include "ftkrp.hpp"

/* 
 * <OUT> double cpderr(<IN> const Ref<const MatrixXd> True_Tensor_X,     
 *                     <IN> const Ref<const MatrixXd> Est_Factor, 
 *                     <IN> const Ref<const MatrixXd> KhatriRao,         
 *                     <IN> MatrixXd                  &Est_Tensor_X, 
 *                     <IN> const Ref<const VectorXi> Tensor_Dimensions, 
 *                     <IN> const int                 current_mode)
 * 
 * Description: Computes cpderr: || True_Tensor_X - Est_Tensor_X ||_2 (used as quality measure for CPD).
 */

inline double cpderr(const Ref<const MatrixXd> True_Tensor_X, const Ref<const MatrixXd> Est_Factor, const Ref<const MatrixXd> KhatriRao,
                     MatrixXd &Est_Tensor_X, const Ref<const VectorXi> Tensor_Dimensions, const int current_mode, const unsigned int num_threads)
{
    // Est_Tensor_X = Est_Factor * KhatriRao.transpose();
    ftkrp(Est_Factor, KhatriRao, Est_Tensor_X, Tensor_Dimensions, current_mode, num_threads);

    return (True_Tensor_X - Est_Tensor_X).norm();
}

#endif