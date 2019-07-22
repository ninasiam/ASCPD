#include "cpd_functions.h"
#include <iostream>

//test for git
using namespace Eigen;

void rand_shuffle_indices(const VectorXi &dim_size, MatrixXi &rand_indices, int factor)
{
    int set_of_rows[dim_size(factor)];
    // Generate random indices
    for (int row = 0; row < dim_size(factor); row++)
    {
        set_of_rows[row] = row;
    }

    std::random_shuffle(&set_of_rows[0], &set_of_rows[dim_size(factor)]);

    for (int row = 0; row < dim_size(factor); row++)
    {
        rand_indices(factor, row) = set_of_rows[row];
    }
    // std::cout << "\n\n\n\n" << rand_indices << "\n";
}

void create_subfactors(const MatrixXd &Factor, MatrixXd &subFactor, const MatrixXd &Matr_Tensor, MatrixXd &Matr_subTensor, 
                        const VectorXi &dim_size, const VectorXi &block_size, MatrixXi &rand_indices, MatrixXi &B_cal, int iter, int factor)
{
    factor-=1;

    // Implementation #1 -- Works only for cubic tensor

    // int Q_n = (int)(dim_size(factor) / block_size(factor));

    // int l_n = 1 + ((iter-1) % Q_n);

    // Case: Reshuffle
    // if(l_n - 1 == 0 && factor == 0){
    // for(int factor_i=0; factor_i<3; factor_i++)
    //     rand_shuffle_indices(dim_size, rand_indices, factor_i);
    // }
    // Create block of indices
    // if(factor == 0){
    //     for(int factor_i=0; factor_i<3; factor_i++)
    //         B_cal.row(factor_i) = rand_indices.block(factor_i, (l_n - 1) * block_size(factor_i), 1, l_n * block_size(factor_i) - 1);
    // }

    // Implementation #2
    if(factor == 0){
        for(int factor_i=0; factor_i<3; factor_i++){
            int Q_n = (int)(dim_size(factor_i) / block_size(factor_i));
            int l_n = 1 + ((iter - 1) % Q_n);
            // Case: Reshuffle
            if(l_n - 1 == 0){    
                rand_shuffle_indices(dim_size, rand_indices, factor_i);
            }
            // Create block of indices
            B_cal.row(factor_i) = rand_indices.block(factor_i, (l_n - 1) * block_size(factor_i), 1, l_n * block_size(factor_i) - 1);
        }
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
        numa_run_on_node(0);
        #pragma omp parallel for schedule(dynamic) \
			  					 default(none) \
								 shared(block_size, dim_size, B_cal, Matr_Tensor, Matr_subTensor)

        for (int k = 0; k < block_size(2); k++) // K: slices 
        {
            for (int j = 0; j < block_size(1); j++) // J: cols
            { 
                for (int i = 0; i < block_size(0); i++) // I: rows
                {
                    Matr_subTensor(i, (k * block_size(1)) + j) = Matr_Tensor(B_cal(0, i), B_cal(1, j) + (B_cal(2, k) * dim_size(1)));
                }
            }
        }
    }
    else if(factor==1) // factor B
    {   
        numa_run_on_node(0);
        #pragma omp parallel for schedule(dynamic) \
                                 default(none) \
                                 shared(block_size,dim_size,B_cal, Matr_Tensor, Matr_subTensor)

        for (int j = 0; j < block_size(0); j++) // I: cols
        {
            for (int k = 0; k < block_size(2); k++) // K: slices
            {
                for (int i = 0; i < block_size(1); i++) // J: rows
                {
                    Matr_subTensor((k * block_size(1)) + i, j) = Matr_Tensor((B_cal(2, k) * dim_size(1)) + B_cal(1, i), B_cal(0, j));
                }
            }
        }
    }
    else if (factor == 2) // factor C
    {
        numa_run_on_node(0);
        #pragma omp parallel for schedule(dynamic) \
                                 default(none) \
                                 shared(block_size, dim_size, B_cal, Matr_Tensor, Matr_subTensor)
                                 
        for (int k = 0; k < block_size(1); k++) // J: slices
        {
            for (int j = 0; j < block_size(0); j++) // I: cols
            {
                for (int i = 0; i < block_size(2); i++) // K: rows
                {
                    Matr_subTensor(i, (k * block_size(0)) + j) = Matr_Tensor(B_cal(2, i), B_cal(0, j) + (B_cal(1, k) * dim_size(0)));
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
