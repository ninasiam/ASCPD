#ifndef CREATE_TENSOR_HPP
#define CREATE_TENSOR_HPP

#include "master_lib.hpp"
#include "generateTensorAtRuntime.hpp"
// #include "khatri_rao_product.hpp"
#include "FullKhatriRaoProduct.hpp"
#include "ftkrp.hpp"
#include "omp_lib.hpp"

/*--+ Create Tensor using true Factors A_{1}...A_{TNS_ORDER} +--*/
/* 
 * void createTensor(<IN>  const int                        tensor_order, 
 *                   <IN>  const Ref<const VectorXi>        tensor_dims, 
 *                   <IN>  int                              tensor_rank, 
 *                   <OUT> Eigen::Tensor<double, TNS_ORDER> &True_Tensor, 
 *                   <IN>  std::array<MatrixXd, TNS_ORDER>  &True_Factors)
 * 
 * Description: Generates Tensor from input known Factors and Rank.
 * 
 * param tensor_order    : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims     : is the vector containing the tensor's dimensions,
 * param tensor_rank     : is the rank of input Tensor X,
 * param True_Tensor     : is the output Tensor X of type Tensor made from input Factors,
 * param True_Factors    : contains all input Factors.
 */

template <int TNS_SIZE, std::size_t TNS_ORDER>
void createTensor(const int tensor_order, const Ref<const VectorXi> tensor_dims, int tensor_rank,
                  Eigen::Tensor<double, TNS_SIZE> &True_Tensor, std::array<MatrixXd, TNS_ORDER> &True_Factors)
{
    generateTensorAtRuntime(tensor_dims, True_Tensor);
    int dim = tensor_dims.prod() / tensor_dims(0);

    MatrixXd Khatri_Rao = MatrixXd::Zero(tensor_rank, dim);

    // int mode_N = tensor_order - 1;

    #ifdef FACTORS_ARE_TRANSPOSED
        // Khatri_Rao.block(0, 0, True_Factors[mode_N].cols(), tensor_rank) = True_Factors[mode_N];

        // for (int ii = 0, rows = True_Factors[mode_N].rows(), curr_dim = mode_N - 1; ii < tensor_order - 2; ii++, curr_dim--)
        // {
        //     if (curr_dim == 0)
        //     {
        //         curr_dim = curr_dim - 1;
        //     }
        //     Khatri_Rao_Product(Khatri_Rao, True_Factors[curr_dim], Khatri_Rao, rows, tensor_rank, get_num_threads());

        //     rows = rows * True_Factors[curr_dim].rows();
        // }
        FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, 0, True_Factors, Khatri_Rao, get_num_threads());
        // True_Tensor_Mat[mode] = True_Factors[mode] * Khatri_Rao.transpose();
        MatrixXd MatricizedTensor = MatrixXd::Zero(tensor_dims(0), dim);

        ftkrp(True_Factors[0], Khatri_Rao, MatricizedTensor, tensor_dims, 0, get_num_threads());

        Eigen::Tensor<double, 2>::Dimensions two_dim(tensor_dims(0), dim);

        // std::cout << True_Tensor << std::endl;
        True_Tensor.reshape(two_dim) = TensorMap<Eigen::Tensor<double, 2>>(MatricizedTensor.data(), two_dim);
    #else
        // Khatri_Rao.block(0, 0, True_Factors[mode_N].rows(), tensor_rank) = True_Factors[mode_N];

        // for (int ii = 0, rows = True_Factors[mode_N].rows(), curr_dim = mode_N - 1; ii < tensor_order - 2; ii++, curr_dim--)
        // {
        //     if (curr_dim == 0)
        //     {
        //         curr_dim = curr_dim - 1;
        //     }
        //     Khatri_Rao_Product(Khatri_Rao, True_Factors[curr_dim], Khatri_Rao, rows, tensor_rank, get_num_threads());

        //     rows = rows * True_Factors[curr_dim].rows();
        // }
        FullKhatriRaoProduct(tensor_order, tensor_dims, tensor_rank, 0, True_Factors, Khatri_Rao, get_num_threads());

        // True_Tensor_Mat[mode] = True_Factors[mode] * Khatri_Rao.transpose();
        MatrixXd MatricizedTensor = MatrixXd::Zero(tensor_dims(0), dim);

        ftkrp(True_Factors[0], Khatri_Rao, MatricizedTensor, tensor_dims, 0, get_num_threads());

        Eigen::Tensor<double, 2>::Dimensions two_dim(tensor_dims(0), dim);

        // std::cout << True_Tensor << std::endl;
        True_Tensor.reshape(two_dim) = TensorMap<Eigen::Tensor<double, 2>>(MatricizedTensor.data(), two_dim);
    #endif

    // std::cout << True_Tensor << std::endl;
    // Eigen::Tensor<double, TNS_ORDER>::Dimensions new_dims;
    // for(int dim_i=0; dim_i<tensor_order; dim_i++)
    // {
    //     new_dims[dim_i] = tensor_dims(dim_i);
    // }
    // True_Tensor.reshape(new_dims);
    // std::cout << "ok" << std::endl;
}


/*--+ Create Tensor using Matricized Tensor X_{1} +--*/
/* 
 * void createTensor(<IN>  const int                        tensor_order, 
 *                   <IN>  const Ref<const VectorXi>        tensor_dims, 
 *                   <OUT> Eigen::Tensor<double, TNS_ORDER> &True_Tensor, 
 *                   <IN>  std::array<MatrixXd, TNS_ORDER>  &MatricizedTensor)
 * 
 * Description: Generates Tensor from input mode-1 tensor matricization.
 * 
 * param tensor_order     : is the order of input Tensor X,
 *        NOTE! tensor_order is redundant. This variable is equal to TNS_ORDER and can be removed in a future version.
 * param tensor_dims      : is the vector containing the tensor's dimentions,
 * param tensor_rank      : is the rank of input Tensor X,
 * param True_Tensor      : is the output Tensor X of type Tensor created from the mode-1 matricization,
 * param MatricizedTensor : is the Matricized input Tensor.
 */
template <int TNS_SIZE>
void createTensor(const int tensor_order, const Ref<const VectorXi> tensor_dims, int tensor_rank,
                  Eigen::Tensor<double, TNS_SIZE> &True_Tensor, Ref<MatrixXd> MatricizedTensor)
{
    generateTensorAtRuntime(tensor_dims, True_Tensor);

    Eigen::Tensor<double, 2>::Dimensions two_dim(MatricizedTensor.rows(), MatricizedTensor.cols());

    // std::cout << True_Tensor << std::endl;
    True_Tensor.reshape(two_dim) = TensorMap<Eigen::Tensor<double, 2>>(MatricizedTensor.data(), two_dim);

}

#endif