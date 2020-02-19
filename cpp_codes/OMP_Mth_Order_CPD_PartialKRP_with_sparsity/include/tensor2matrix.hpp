#ifndef TENSOR2MATRIX_HPP
#define TENSOR2MATRIX_HPP

#include "master_lib.hpp"

/* 
 * void tensor2matrix(<IN>  const Eigen::Tensor<double, TNS_ORDER> TNS, 
 *                    <OUT> Eigen::MatrixXd                        &MATRIX)
 * 
 * Description: Maps a Tensor to the corresponding Matrix.
 */
template <int TNS_SIZE>
void tensor2matrix(Eigen::Tensor<double, TNS_SIZE> TNS, Eigen::MatrixXd &MATRIX, const Eigen::VectorXi tensor_dims, const int current_mode)
{

    Eigen::Tensor<double, 2> MatricizedTNS;

    // Eigen::Tensor<double, TNS_ORDER>::Dimensions new_dims;
    std::array<int, TNS_SIZE> new_dims_idx;
    // Eigen::array<ptrdiff_t, TNS_ORDER> new_dims;

    for(int dim_idx = 0; dim_idx < TNS_SIZE; dim_idx++)
    {
        new_dims_idx[dim_idx] = dim_idx; // dim_idx = {0, 1, 2, ... , n, ..., N}
    }

    new_dims_idx[0] = current_mode;      // dim_idx = {n, 0, 1, ..., n-1, n+1, ... , N}
    // std::cout << new_dims_idx[0];

    for(int dim_idx = 1; dim_idx < current_mode + 1; dim_idx++)
    {
        new_dims_idx[dim_idx] = dim_idx - 1;    
        // std::cout << "\t" << new_dims_idx[dim_idx];
    }
    // std::cout << std::endl << std::endl;
    
    Eigen::Tensor<double, 2>::Dimensions two_dim(MATRIX.rows(), MATRIX.cols());
    
    // Eigen::Tensor<double, TNS_ORDER> TNS_copy = TNS;
    // // TNS_copy.resize(new_dims);
    // TNS_copy = TNS.shuffle(new_dims_idx);
    // MatricizedTNS = (TNS_copy).reshape(two_dim);

    MatricizedTNS = (TNS.shuffle(new_dims_idx)).reshape(two_dim);

    // Map Matricized Tensor <TNS> to Matrix <MATRIX>
    MATRIX = Eigen::Map<Eigen::MatrixXd>(MatricizedTNS.data(), MATRIX.rows(), MATRIX.cols());
}

#endif