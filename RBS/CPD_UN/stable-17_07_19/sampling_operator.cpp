#include "cpd_functions.h"
#include <iostream>

using namespace Eigen;

void rand_shuffle_indices(const MatrixXd &Factor, MatrixXi &rand_indices, int factor)
{
    int set_of_rows[Factor.rows()];
    // Generate random indices
    for (int row = 0; row < Factor.rows(); row++)
    {
        set_of_rows[row] = row;
    }

    std::random_shuffle(&set_of_rows[0], &set_of_rows[int(Factor.rows())]);

    for (int row = 0; row < Factor.rows(); row++)
    {
        rand_indices(factor, row) = set_of_rows[row];
    }
    // std::cout << rand_indices << "\n";
}

void create_subfactors(const MatrixXd &Factor, MatrixXd &subFactor, const MatrixXd &Matr_Tensor, MatrixXd &Matr_subTensor, const VectorXi &block_size, MatrixXi &rand_indices, MatrixXi &B_cal, int iter, int factor)
{
    factor-=1;
    
    int Q_n = (int)Factor.rows() / block_size(factor);

    int l_n = 1 + ((iter-1) % Q_n);

    // Case: Reshuffle
    if(l_n - 1 == 0 && factor == 0){
        for(int factor_i=0; factor_i<3; factor_i++)
            rand_shuffle_indices(Factor, rand_indices, factor_i);
    }

    // Create block of indices
    if(factor == 0){
        for(int factor_i=0; factor_i<3; factor_i++)
            B_cal.row(factor_i) = rand_indices.block(factor_i, (l_n - 1) * block_size(factor_i), 1, l_n * block_size(factor_i) - 1);
    }
    // Generate SubFactors
    for (int row = 0; row < block_size(factor); row++){
        subFactor.row(row) = Factor.row(B_cal(factor, row));
    }
    // Generate Matricized Sub Tensor
    // int dim[3];
    // for(int dim_i=0; dim_i<3; dim_i++){
    //     dim[dim_i] = (dim_i + factor) % 3;
    // }
    if(factor==0)    // factor A
    {
        for (int i = 0; i < block_size(0); i++)
        {
            for (int j = 0; j < block_size(1); j++)
            {
                for (int k = 0; k < block_size(2); k++)
                {
                    Matr_subTensor(i, (k * block_size(1)) + j) = Matr_Tensor(B_cal(0, i), B_cal(1, j) + (B_cal(2, k) * rand_indices.cols()));

                    // std::cout << Matr_Tensor(B_cal(dim[0], i), B_cal(dim[1], j) + (B_cal(dim[2], k) * rand_indices.cols())) << "\t" << i << " " << j << " " << k << "\t"
                    //           << B_cal(dim[0], i) << " " << B_cal(dim[1], j) << " " << B_cal(dim[2], k) << "\t" << block_size(dim[2]) << "\n\n";
                }
            }
        }
    }
    else if(factor==1) // factor B
    {
        for (int i = 0; i < block_size(1); i++)
        {
            for (int j = 0; j < block_size(0); j++)
            {
                for (int k = 0; k < block_size(2); k++)
                {
                    Matr_subTensor((k * block_size(2)) + i, j) = Matr_Tensor((B_cal(2, k) * rand_indices.cols()) + B_cal(1, i), B_cal(0, j));
                }
            }
        }
    }
    else if (factor == 2) // factor C
    {
        for (int i = 0; i < block_size(2); i++)
        {
            for (int j = 0; j < block_size(0); j++)
            {
                for (int k = 0; k < block_size(1); k++)
                {
                    Matr_subTensor(i, (k * block_size(0)) + j) = Matr_Tensor(B_cal(2, i), B_cal(0, j) + (B_cal(1, k) * rand_indices.cols()));

                    // std::cout << Matr_Tensor(B_cal(dim[0], i), B_cal(dim[1], j) + (B_cal(dim[2], k) * rand_indices.cols())) << "\t" << i << " " << j << " " << k << "\t"
                    //           << B_cal(dim[0], i) << " " << B_cal(dim[1], j) << " " << B_cal(dim[2], k) << "\t" << block_size(dim[2]) << "\n\n";
                }
            }
        }
    }
    else
    {
        std::cout << "factor value out of range!" << std::endl;
        exit(1);
    }
    
}

void merge_Factors(MatrixXd &Factor, const MatrixXd &subFactor, const VectorXi &block_size, MatrixXi &B_cal, int factor)
{
    factor -= 1;
    for (int row = 0; row < block_size(factor); row++)
    {
        Factor.row(B_cal(factor, row)) = subFactor.row(row);
    }
}
