#ifndef SAMPLING_FUNS_HPP
#define SAMPLING_FUNS_HPP

#include "master_library.hpp"
#include "khatri_rao_prod.hpp"

namespace v1
{
        inline void Sampling_Operator(int order, VectorXi block_size, VectorXi dims,
                                      VectorXi &F_n, int &factor)
    {

        // -----------------------> Choose Among Factors to optimize <-------------------
        int n;
        int J_n;
        int MAX_idx = order;

                            
        n  = rand() % MAX_idx;      

        if(n == 0)                                              // Factor A
        {   
            J_n = dims(1)*dims(2);
            // kr_idx(0) = 2;
            // kr_idx(1) = 1;
        }
        if(n == 1)                                              // Factor B
        {
            J_n = dims(0)*dims(2);
            // kr_idx(0) = 2;
            // kr_idx(1) = 0;
        }
        if(n == 2)                                              // Factor C
        {
            J_n = dims(0)*dims(1);
            // kr_idx(0) = 1;
            // kr_idx(1) = 0;
        }

        factor = n;                                             // Selected factor
        
        //----------------------> Generate indices <------------------------------------
        VectorXi indices(J_n,1);
        for(int i=0; i < J_n; i++)
        {
            indices(i) = i; 
        }
        random_device rd;
        mt19937 g(rd());
    
        shuffle(indices.data(), indices.data() + J_n, g);

        F_n = indices.head(block_size(factor));
        
        sort(F_n.data(), F_n.data() + block_size(factor));      //sort F_n
        // cout << F_n << endl;
    }

    void Sampling_Sub_Matrices(const VectorXi &F_n, const MatrixXd &X, const MatrixXd &U1, const MatrixXd &U2, 
                               MatrixXd &KhatriRao, MatrixXd &KhatriRao_sub, MatrixXd &X_sub)
    {   
        int J_n = KhatriRao.rows();
        int R = KhatriRao.cols();
        int bz = F_n.size();
        MatrixXd KhatriRao_T(R, J_n);

        Khatri_Rao_Product(U1, U2, KhatriRao);                          // Compute the full Khatri-Rao Product
        
        KhatriRao_T = KhatriRao.transpose();

        for(int col_H = 0; col_H < bz; col_H++)
        {
            KhatriRao_sub.col(col_H) = KhatriRao_T.col(F_n(col_H));     //Create KhatriRao_sub (transpose)
        }
        
        for(int col_X = 0; col_X < bz; col_X++)
        {
            X_sub.col(col_X) = X.col(F_n(col_X));
        }
        
    }
} // end of namespace v1

namespace symmetric //for symmetric tensors
{
    inline void Sample_mode(int TNS_ORDER, int &current_mode)
    {   
        //Choose the factor to be updated
        current_mode  = rand() % TNS_ORDER;  
    }

    inline void Sample_Fibers(double* Tensor_pointer, const VectorXi &tns_dims, const VectorXi &block_size, int current_mode,
                       MatrixXi &sampled_idxs, MatrixXi &factor_idxs, MatrixXd &T_mode) //sample symmetric tensors
    {
        size_t order = block_size.size();
        size_t numCols_reduced = factor_idxs.cols(); // dimensions: block-size x order-1
        size_t numRows_reduced;
        int offset_sum;
       // double* Tensor_pointer = Tensor.data();
        //Initialize true indices
        MatrixXi true_indices(tns_dims(current_mode), order);
        MatrixXi idxs(block_size(current_mode),order);
        VectorXi vector_offset(order,1);
        VectorXi current_vector_offset(order,1);
        VectorXi dims_offset(order - 1,1);
        VectorXi offset(order - 1,1);
        VectorXi index_vec(tns_dims(current_mode),1);


        for(size_t index = 0; index < tns_dims(current_mode); index ++)
        {
            index_vec(index) = index;
        }
        
        //Shuffle true indices
        for(size_t cols_t = 0 ; cols_t < order ; cols_t++)
        {   
            // #if INITIALIZED_SEED
            //     random_device rd;
            //     mt19937 g(rd());
            //     shuffle(index_vec.data(), index_vec.data() + tns_dims(cols_t), g);
            // #endif

            random_shuffle(index_vec.data(), index_vec.data() + tns_dims(cols_t));
            true_indices.col(cols_t) = index_vec;
        }

        // cout << "true_indices \n" << true_indices <<endl;

        //sample blocksize indices for every mode (including current)
        idxs = true_indices.topRows(block_size(current_mode));
        // cout << "sampled idxs \n" << idxs << endl;

        sampled_idxs = idxs;
        //Remove the indices for the current mode

        numRows_reduced = idxs.rows();
        if( current_mode < numCols_reduced )
            idxs.block(0, current_mode, numRows_reduced, numCols_reduced - current_mode) = idxs.rightCols(numCols_reduced - current_mode);

        idxs.conservativeResize(numRows_reduced,numCols_reduced);
        factor_idxs = idxs;
       
        //cout << "factor idxs \n" << factor_idxs << endl;

        //create the offset vector for mode_1
        vector_offset(0) = 1;
        for(size_t dim_idx = 1; dim_idx < order; dim_idx++)
        {
           vector_offset(dim_idx) = vector_offset(dim_idx - 1)*tns_dims(dim_idx -1);
        }

        //create current vector offset (not the first mode)
        if(current_mode != 0)
        {
            current_vector_offset = vector_offset;
            current_vector_offset(0) = vector_offset(current_mode);
            for(size_t dim_idx = 1; dim_idx < current_mode + 1; dim_idx++)
            {
                current_vector_offset(dim_idx) = vector_offset(dim_idx - 1);
            }
            vector_offset = current_vector_offset;
        }
        //cout << "vector_offset \n" << vector_offset << endl;

        //sample the fibers
        dims_offset = vector_offset.tail(order - 1);   //offset for each mode (truncate the first element which correspond to the current mode)
        for(size_t fiber = 0; fiber < block_size(current_mode); fiber++)
        {
            //create the offset for each fiber   
            offset = dims_offset.cwiseProduct(factor_idxs.row(fiber).transpose());
            offset_sum = offset.sum();

            for (int el = 0; el < tns_dims(current_mode); el++)
            {
                 T_mode(el,fiber) = Tensor_pointer[vector_offset(0)*el + offset_sum];  //fibers as columns of the matricization
            }
           
        }
    }

    template <std::size_t  TNS_ORDER>
    inline void Sample_KhatriRao(const int &current_mode, const size_t &R, const MatrixXi &sampled_indices, std::array<MatrixXd, TNS_ORDER> &Factors, MatrixXd &KR_sampled)
    {
        size_t order = Factors.size();
        size_t kr_s_rows = sampled_indices.rows();
                                                                                  //set the KR_sampled to ones (for the product down)
        KR_sampled.setOnes(kr_s_rows,R);                                                       

        //For every row of Khatri-Rao (sampled)
        for(int kr_s_row = 0; kr_s_row < kr_s_rows; kr_s_row++)                  //for every row of the sampled kr (NOT size_int because current_mode is of type int)
        {
            for(int factor = order - 1; factor > -1; factor--)                   //for each factor (expept the current mode)
            {
                if( factor != current_mode)
                {   
                    
                    KR_sampled.row(kr_s_row) = KR_sampled.row(kr_s_row).cwiseProduct(Factors[factor].row(sampled_indices(kr_s_row, factor)));
                    //KR_sampled.row(kr_s_row) *= Factors[factor].row(sampled_indices(kr_s_row, factor));
                }

            }
            
        }

        

    }
} // end of namespace symmetric


// namespace v2
// {   
//     template <typename T>
//     void Sampling_fibers(const T &Tensor, int mode, VectorXi &tns_dims, VectorXi &block_size,
//                          MatrixXd idxs, MatrixXd factor_idxs, MatrixXd T_mode)
//     {
//         int order = block_size.size();

//         //Initialize true indices
//         MatrixXd true_indices(tns_dims(0), order);
//         VectorXi index_vec(tns_dims(0),1);

//         for(int index = 0; index < tns_dims(0); index ++)
//         {
//             index_vec(index) = index;
//         }
        
//         //Shuffle true indices
//         for(int cols_t = 0 ; cols_t < order ; cols_t++)
//         {
//             random_device rd;
//             mt19937 g(rd());
//             true_indices.col(cols_t) = index_vec;
//             shuffle(true_indices.col(cols_t).data(), true_indices.col(cols_t).data() + tns_dims(cols_t), g);
//             idxs.col(cols_t) = true_indices.col().head(block_size(cols_t));
//         }
        
//         //Create Matricization
//         if( mode == 0) 
//         {
//             for(int col_T = 0; col_T < block_size(mode); col_T++ )
//             {
//                 for(int row_T = 0; row_T < tns_dims(mode); row_T++)
//                 {
//                     T_mode(row_T, col_T) = Tensor(row_T, idxs(row_T, 1), idxs(row_T, 2));
//                 }
                
//             }
//         }
//         if( mode == 1) 
//         {
//             for(int col_T = 0; col_T < block_size(mode); col_T++ )
//             {
//                 for(int row_T = 0; row_T < tns_dims(mode); row_T++)
//                 {
//                     T_mode(row_T, col_T) = Tensor( idxs(row_T, 0), row_T, idxs(row_T, 2));
//                 }
                
//             }
//         }
//         if( mode == 2) 
//         {
//             for(int col_T = 0; col_T < block_size(mode); col_T++ )
//             {
//                 for(int row_T = 0; row_T < tns_dims(mode); row_T++)
//                 {
//                     T_mode(row_T, col_T) = Tensor(idxs(row_T, 0), idxs(row_T, 1), row_T);
//                 }
                
//             }
//         }


         

        
//     }
    
// } // end of namespace v2

#endif //end if