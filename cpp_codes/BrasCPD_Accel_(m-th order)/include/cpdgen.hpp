#ifndef CPDGEN_HPP
#define CPDGEN_HPP

#include "master_library.hpp"


auto Matrix_to_Tensor( Eigen::MatrixXd matrix, int dim, int R)
{
    
    return Eigen::TensorMap<Eigen::Tensor<const double, 2>>(matrix.data(), {dim, R} );
}

template <std::size_t  TNS_ORDER>
void CpdGen( VectorXi tns_dims, std::array<MatrixXd, TNS_ORDER> Factors, int const R, Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > &Est_Tensor)
{
    using Factor_array = std::array<MatrixXd, TNS_ORDER>;
    Eigen::Tensor< double, static_cast<int>(TNS_ORDER) > Tmp_Tensor;
    Eigen::Tensor<double, 2> tmp_tensor_mat;
    std::array<int, TNS_ORDER> dims;                                    // For the Identity Tensor

    int i = 0;
    std:: size_t linear_index_2;
    constexpr int w = 1;
    int dim_ident;

    Eigen::VectorXi linear_index(TNS_ORDER);
    Eigen::VectorXi offset(TNS_ORDER);

    std::array<Eigen::IndexPair<int>, 1> prod_dims;

    std::fill(dims.begin(), dims.end(), R);

    // Offset for the identity tensor
    offset(0) = 1;
    for(size_t j = 1; j < TNS_ORDER; j++)
    {
        offset(j) = dims[j]*offset[j-1];
    }

    //Form Identity tensor
    Est_Tensor.resize(dims);
    Est_Tensor.setZero();
 
    dim_ident =  dims[0];
    for(int i = 0; i < dim_ident; i++)
    {   
        linear_index = i*offset;
        linear_index_2 = linear_index.sum();
        Est_Tensor(linear_index_2) = 1;
    }
    
    int dim_idx =  TNS_ORDER - 1;
    for(typename Factor_array::reverse_iterator it = Factors.rbegin(); it != Factors.rend(); ++it )
    {   
        tmp_tensor_mat = Matrix_to_Tensor((*it), tns_dims(dim_idx) , R);
        // cout << tmp_tensor_mat << endl;
        prod_dims = { Eigen::IndexPair<int>(w,i) };
        Tmp_Tensor = tmp_tensor_mat.contract(Est_Tensor, prod_dims);
        Est_Tensor = Tmp_Tensor;
        dim_idx--;
        i++;
    }
}

#endif