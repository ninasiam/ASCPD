#ifndef CPDGEN_HPP
#define CPDGEN_HPP

#include "master_library.hpp"


template <size_t  TNS_ORDER>
void CpdGen( VectorXi tns_dims, std::array<MatrixXd, TNS_ORDER> &Factors, int const R, Eigen::Tensor< double, TNS_ORDER > &Est_Tensor)
{
    using Factor_array = std::array<MatrixXd, TNS_ORDER>;

    Eigen::Tensor< double, TNS_ORDER > Tmp_Tensor;

    std::array<int, TNS_ORDER> dims;
    
    int i = 0;
    constexpr int w = 1;
    const int dim_ident = tns_dims[0];
    Eigen::VectorXi linear_index(1, TNS_ORDER);
    Eigen::VectorXi offset(1, TNS_ORDER);

    std::array<Eigen::IndexPair<int>, 1> prod_dims;

    std::fill(dims.begin(), dims.end(), R);

    offset(0) = 1;
    for(int j = 1; j < TNS_ORDER; j++)
    {
        offset(j) = tns_dims(j)*tns_dims(j-1);
    }

    //Identiy tensor
    for(int i = 0; i < dim_ident, i++)
    {   
        linear_index = i*offset;
        linear_index = linear_index.sum();
        Est_Tensor(linear_index) = 1;
    }

    for(typename Factor_array::reverse_iterator it = Factors.rbegin(); it != Factors.rend(); ++it )
    {
        prod_dims = { Eigen::IndexPair<int>(w,i) };
        Tmp_Tensor = (*it).contract(Est_Tensor, prod_dims);
        Est_Tensor = Tmp_Tensor;
        i++;
    }
}

#endif