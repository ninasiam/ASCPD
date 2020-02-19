// #ifndef CREATE_TENSOR_HPP
// #define CREATE_TENSOR_HPP

// #include "master_lib.hpp"
// #include "khatri_rao_product.hpp"
// #include "ftkrp.hpp"

// /* 
//  * void create_tensor(<IN> const int tensor_order, <IN> const Ref<const VectorXi> tensor_dims, 
//  *                    <IN> int tensor_rank,        <IN> std::array<MatrixXd, TNS_ORDER> &True_Factors, 
//  *                    <OUT> std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat)
//  * 
//  * Description: Generates a matricised tensor of order "tensor_order" and rank "tensor_rank" using synthetic factors.
//  */

// inline void create_tensor(const int tensor_order, const Ref<const VectorXi> tensor_dims, int tensor_rank,
//                           std::array<MatrixXd, TNS_ORDER> &True_Factors, std::array<MatrixXd, TNS_ORDER> &True_Tensor_Mat)
// {
//     for (int mode = 0; mode < tensor_order; mode++)
//     {
//         // dim = I_(1) * ... * I_(mode-1) * I_(mode+1) * ... * I_(N)
//         int dim = tensor_dims.prod() / tensor_dims(mode);

//         MatrixXd Khatri_Rao = MatrixXd::Zero(dim, tensor_rank);

//         int mode_N = tensor_order - 1;
//         if (mode == mode_N)
//         {
//             mode_N = mode - 1;
//         }

//         Khatri_Rao.block(0, 0, True_Factors[mode_N].rows(), tensor_rank) = True_Factors[mode_N];

//         for (int ii = 0, rows = True_Factors[mode_N].rows(), curr_dim = mode_N - 1; ii < tensor_order - 2; ii++, curr_dim--)
//         {
//             if (curr_dim == mode)
//             {
//                 curr_dim = curr_dim - 1;
//             }
//             Khatri_Rao_Product(Khatri_Rao, True_Factors[curr_dim], Khatri_Rao, rows, tensor_rank);

//             rows = rows * True_Factors[curr_dim].rows();
//         }
//         // True_Tensor_Mat[mode] = True_Factors[mode] * Khatri_Rao.transpose();
//         True_Tensor_Mat[mode] = MatrixXd::Zero(tensor_dims(mode), dim);
        
//         ftkrp(True_Factors[mode], Khatri_Rao, True_Tensor_Mat[mode], tensor_dims, mode);
//     }
// }

// #endif