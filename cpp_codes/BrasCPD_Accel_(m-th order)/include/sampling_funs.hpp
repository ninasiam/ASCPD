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
        // current_mode = 2;
    }

    inline void Sample_Fibers(double* Tensor_pointer, const VectorXi &tns_dims, const VectorXi &block_size, int current_mode,
                       MatrixXi &sampled_idxs, MatrixXi &factor_idxs, MatrixXd &T_mode) //sample symmetric tensors
    {
        size_t order = block_size.size();
        size_t numCols_reduced = factor_idxs.cols(); // dimensions: block-size x order-1
        size_t numRows_reduced;
        size_t offset_sum;

        //Initialize true indices
        MatrixXi true_indices(tns_dims(current_mode), order);
        MatrixXi idxs(block_size(current_mode),order);
        VectorXi vector_offset(order,1);
        VectorXi current_vector_offset(order,1);
        VectorXi dims_offset(order - 1,1);
        VectorXi offset(order - 1,1);
        VectorXi index_vec(tns_dims(current_mode),1);
        VectorXi idxs_full = tns_dims;

        
        // for(size_t index = 0; index < tns_dims(current_mode); index ++)
        // {
        //     index_vec(index) = index;
        // }
        // index_vec.replicate(tns_dims(current_mode),1);
        // cout << "index_vec" << index_vec << endl;
        // //Shuffle true indices
        // for(size_t cols_t = 0 ; cols_t < order ; cols_t++)
        // {   
        //     // #if INITIALIZED_SEED
        //     //     random_device rd;
        //     //     mt19937 g(rd());
        //     //     shuffle(index_vec.data(), index_vec.data() + tns_dims(cols_t), g);
        //     // #endif

        //     random_shuffle(index_vec.data(), index_vec.data() + tns_dims(cols_t));
        //     true_indices.col(cols_t) = index_vec;
        // }
        // cout << "true_idxs" << true_indices << endl;
        
        
        //sample blocksize indices for every mode (including current)
        for(int tuple = 0; tuple < block_size(current_mode); tuple++)
        {
            for(int col = 0; col < order; col++)
            {
                idxs(tuple, col) = rand() % tns_dims(col);
            }

        }
        // idxs = true_indices.topRows(block_size(current_mode));

        sampled_idxs = idxs;
        // cout << "sampled_idxs" << sampled_idxs << endl;
        //Remove the indices for the current mode
        numRows_reduced = idxs.rows();
        if( current_mode < numCols_reduced )
            idxs.block(0, current_mode, numRows_reduced, numCols_reduced - current_mode) = idxs.rightCols(numCols_reduced - current_mode);

        idxs.conservativeResize(numRows_reduced,numCols_reduced);
        factor_idxs = idxs;
       
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

        //sample the fibers
        dims_offset = vector_offset.tail(order - 1);   //offset for each mode (truncate the first element which correspond to the current mode)
        for(size_t fiber = 0; fiber < block_size(current_mode); fiber++)
        {
            //create the offset for each fiber   
            offset = dims_offset.cwiseProduct(factor_idxs.row(fiber).transpose());
            offset_sum = offset.sum();

            for (size_t el = 0; el < tns_dims(current_mode); el++)
            {
                 T_mode(el,fiber) = Tensor_pointer[vector_offset(0)*el + offset_sum];  //fibers as columns of the matricization
            }
           
        }
    }

    template <std::size_t  TNS_ORDER>
    inline void Sample_KhatriRao(const int &current_mode, const int &R, const MatrixXi &sampled_indices, const std::array<MatrixXd, TNS_ORDER> &Factors, MatrixXd &KR_sampled)
    {
        int order = Factors.size();
        int kr_s_rows = sampled_indices.rows();
      
        KR_sampled.setOnes(kr_s_rows,R);                                                       

        //For every row of Khatri-Rao (sampled)
        for(int kr_s_row = 0; kr_s_row < kr_s_rows; kr_s_row++)                  //for every row of the sampled kr (NOT size_int because current_mode is of type int)
        {
            for(int factor = order - 1; factor > -1; factor--)                   //for each factor (except the current mode)
            {
                if( factor != current_mode)
                {   
                    
                    KR_sampled.row(kr_s_row) = KR_sampled.row(kr_s_row).cwiseProduct(Factors[factor].row(sampled_indices(kr_s_row, factor)));
                    //KR_sampled.row(kr_s_row) *= Factors[factor].row(sampled_indices(kr_s_row, factor));
                }

            }
            
        }

        

    }
} 

namespace sorted // sorted namespace (The indices are now sorted BUT we need to include khatri-rao to this implementation)
{   
    template <int TNS_ORDER>
    bool sortBasedCols(const std::vector<int> &v1, const std::vector<int> &v2)
    {   
        // int order =  v1.size() + 1;
        bool final_expr {false};
        std::array<bool,TNS_ORDER - 1> expr;
        for(int i = TNS_ORDER - 2; i >= 0; i--)
        {
            if(i < TNS_ORDER - 2)
            {
                expr[i] = (v1[i+1] == v2[i+1]) && (v1[i] < v2[i]);
                final_expr = final_expr || expr[i];

                if(!final_expr)
                {
                    return final_expr;
                }
            }
            else
            {
                expr[i] = v1[i] < v2[i];
                final_expr = final_expr || expr[i];
            }
        }

        return final_expr;
    }

    // template <int TNS_ORDER>
    // bool sortBasedCols(const std::vector<int> &v1, const std::vector<int> &v2)
    // {
    //     return v1[TNS_ORDER - 1] < v2[TNS_ORDER - 1];
    // }

    inline void Sample_mode(int TNS_ORDER, int &current_mode)
    {   
        //Choose the factor to be updated
        current_mode  = rand() % TNS_ORDER;  
    }

    template <int TNS_ORDER>
    std::vector<std::vector<int>> Sample_fibers(double* Tensor_pointer, const VectorXi &tns_dims, const VectorXi &block_size, int current_mode,
                       MatrixXi &sampled_idxs, MatrixXd &T_mode)
    {   
        int idx_val;
        int order = block_size.size();
        
        std::vector<int> vector_offset;
        std::vector<int> current_vector_offset;
        std::vector<std::vector<int> > fiber_idxs;                                  // vector of vectors, 2D vector
        std::vector<size_t> offset_fiber;
        offset_fiber.resize(block_size(current_mode), 1);
        fiber_idxs.resize(block_size(current_mode), std::vector<int>(TNS_ORDER - 1));   // resize it, in order to index it
        vector_offset.resize(order,1);
        current_vector_offset.resize(order,1);
        // idxs: sample blocksize tuple of indices for every mode (including current)
        // fiber_idxs: same structure as factor_idxs but it is now a vector of vectors
        for(int tuple = 0; tuple < block_size(current_mode); tuple++)
        {
            for(int col = 0, inner_col = 0; col < order; col++)
            {   
                idx_val = rand() % tns_dims(col);
                sampled_idxs(tuple, col) = idx_val;

                if(current_mode == col)
                {   
                    inner_col = 1;
                    continue;
                }
                else
                {   
                    if(inner_col == 1)
                    {
                        fiber_idxs[tuple][col - 1] = idx_val;
                    }
                    else
                    {
                        fiber_idxs[tuple][col] = idx_val;
                    }

                }
            }
        }

        // Number of rows; 
        int m = fiber_idxs.size();  
        int n = fiber_idxs[0].size(); 


        // Displaying the 2D vector before sorting 
        // cout << "The Matrix before sorting 1st row is:\n"; 
        // for (int i=0; i<m; i++) 
        // { 
        //     for (int j=0; j<n ;j++) 
        //         cout << fiber_idxs[i][j] << " "; 
        //     cout << endl; 
        // } 


        // Sort the fiber_idxs
        std::sort(fiber_idxs.begin(), fiber_idxs.end(), sorted::sortBasedCols<TNS_ORDER>);

        
        // Displaying the 2D vector after sorting 
        // cout << "The Matrix after sorting 1st row is:\n"; 
        // for (int i=0; i<m; i++) 
        // { 
        //     for (int j=0; j<n ;j++) 
        //         cout << fiber_idxs[i][j] << " "; 
        //     cout << endl; 
        // } 

        //create the offset vector for mode_1
        // cout << "SORTING FIBERS START!";

        vector_offset[0] = 1;
        for(size_t dim_idx = 1; dim_idx < order; dim_idx++)
        {
           vector_offset[dim_idx] = vector_offset[dim_idx - 1]*tns_dims[dim_idx -1];
        }

        //create current vector offset (not the first mode)
        if(current_mode != 0)
        {
            current_vector_offset = vector_offset;
            current_vector_offset[0] = vector_offset[current_mode];
            for(size_t dim_idx = 1; dim_idx < current_mode + 1; dim_idx++)
            {
                current_vector_offset[dim_idx] = vector_offset[dim_idx - 1];
            }
            vector_offset = current_vector_offset;
        }

        size_t offset = 0;
        if(current_mode == 0 || current_mode  == 1) // Only for mode 0 and 1
        {
            for(size_t fiber = 0; fiber < block_size(current_mode); fiber++)
            {   
                //create the offset for each fiber
                for (int i=1; i<order; i++)  // it counts from one since vector_offset corresponds to the offset of fiber that will added later
                { 
                    offset += vector_offset[i]*fiber_idxs[fiber][i - 1];  
                }
                // std::cout << "offset = :" << offset << std::endl;
                for (size_t el = 0; el < tns_dims(current_mode); el++)
                {
                    T_mode(el,fiber) = Tensor_pointer[vector_offset[0]*el + offset];  //fibers as columns of the matricization
                }
                offset = 0;
           
            }
        }
        else // fill all collumns of matricization at once
        {   // offset = 0;
            for(size_t fiber = 0; fiber < block_size(current_mode); fiber++)
            {
                //create the offset for each fiber
                for (int i=1; i<order; i++)  // it counts from one since vector_offset corresponds to the offset of fiber that will added later
                { 
                    offset += vector_offset[i]*fiber_idxs[fiber][i - 1];  
                    
                }
                // std::cout << "offset = " << offset << std::endl;
                offset_fiber[fiber] = offset;
                offset = 0;
                // std::cout << "offset_fiber: " << offset_fiber[fiber] << std::endl;
            }
            
            for(size_t el = 0; el < tns_dims(current_mode); el++)
            {   
                for(size_t fiber = 0; fiber < block_size(current_mode); fiber++)
                {
                    T_mode(el,fiber) = Tensor_pointer[vector_offset[0]*el + offset_fiber[fiber]];
                }
            }
        }
        // cout << "\t SORTING FIBERS END!" << endl;
        return fiber_idxs;
    }

    template <std::size_t  TNS_ORDER>
    void Sample_KhatriRao(const int &current_mode, const int &R, const std::vector<std::vector<int>> &sampled_indices, const std::array<MatrixXd, TNS_ORDER> &Factors, MatrixXd &KR_sampled)
    {
        int order = Factors.size();
        int kr_s_rows = sampled_indices.size();;
        int kr_factor;
        KR_sampled.setOnes(kr_s_rows,R);                                                       

        int first_mode = (current_mode == 0)? 1: 0;
        int last_mode = (current_mode == order-1)? order-2 : order-1;

        //For every row of Khatri-Rao (sampled)
        for(int kr_s_row = 0; kr_s_row < kr_s_rows; kr_s_row++)                  //for every row of the sampled kr (NOT size_int because current_mode is of type int)
        {
            for(int factor = order - 1, fiber_factor_idx = order - 2; factor > -1; factor--)                   //for each factor (except the current mode)
            {
                if( factor == current_mode)
                {   
                    continue;            
                }
                else
                {
                    KR_sampled.row(kr_s_row) = KR_sampled.row(kr_s_row).cwiseProduct(Factors[factor].row(sampled_indices[kr_s_row][fiber_factor_idx]));
                    fiber_factor_idx --; 
                }
                

            }
            
        }

        

    }

} // end namespace sorted

#endif //end if