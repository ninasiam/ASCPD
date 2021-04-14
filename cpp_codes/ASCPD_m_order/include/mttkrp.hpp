#ifndef MTTKRP_HPP
#define MTTKRP_HPP

#include <omp.h>
#include "master_library.hpp"
#include "omp_lib.hpp"

/* IN : matricization of tensor, Khatri-rao Product, tns_dims, current mode, number of threads
   OUT: Matricized tensor times Khatri - rao */

namespace v1
{
    inline void mttkrp( const MatrixXd &X_mat, const MatrixXd &KR_p, const VectorXi Tns_dims, int Mode,  const unsigned int n_thrds, MatrixXd &Mttkrp)
    {
        #ifndef EIGEN_DONT_PARALLELIZE
            Eigen::setNbThreads(1);
        #endif

        int rows_X_mat = X_mat.rows();
        // cout << "rows" << rows_X_mat << endl;
        int cols_X_mat = X_mat.cols();
        // cout << "cols" << cols_X_mat << endl;

        // cout << "Kr_rows" << KR_p.rows() << endl;
        VectorXi rest_dims = Tns_dims;
        rest_dims(Mode) = 1;
        int max_dim = rest_dims.maxCoeff();
        int rounds;
        int rounds_sz;
        VectorXi offset;

        int cols_X_mat_full = rest_dims.prod();                         // for the full

        Mttkrp.setZero();

        if( cols_X_mat < cols_X_mat_full)                               // if X_mat_sub has fewer columns (e.g bs(mode))
        {   
            if( cols_X_mat > n_thrds)
                {
                    rounds = n_thrds;                                   // Number of blocks 
                    rounds_sz = cols_X_mat / n_thrds;               // Quot

                    int residual = cols_X_mat % n_thrds;                // Residual, in case the cols of X / n_thrds leaves a residual 

                    if( residual != 0)                                  // in case we have a residual
                    {
                        VectorXi offset_tmp(rounds,1);
                        offset_tmp.setConstant(rounds_sz);              // create offset vector with rounds_sz offset for each block
                        offset_tmp(rounds - 1) += residual;             //in the last one add the residual
                        offset = offset_tmp;

                    }
                    else
                    {
                        VectorXi offset_tmp(rounds,1);
                        offset_tmp.setConstant(rounds_sz);              // create offset vector with rounds_sz offset for each block
                        offset = offset_tmp;    
                        // cout << "offset" << offset << endl;                
                    }
            }
            else
            {
                    rounds = 1;      
                    rounds_sz = cols_X_mat;                                    // Number of blocks in case the bs is equal to n_threads
                    VectorXi offset_tmp(rounds,1);
                    offset_tmp.setConstant(rounds_sz);
                    offset = offset_tmp;
            }
            

        }
        else //for the full case
        {   
            rounds = max_dim;
            rounds_sz = cols_X_mat_full/max_dim;
            VectorXi offset_tmp(rounds,1);
            offset_tmp.setConstant(rounds_sz);
            offset = offset_tmp;         
            
        }
        
        #pragma omp parallel for reduction(sum: Mttkrp) default(shared) num_threads(n_thrds)
        for(int block = 0; block < rounds; block++ )
        {
            Mttkrp.noalias() += X_mat.block(0, block * rounds_sz, Mttkrp.rows(), offset(block)) * KR_p.block(block * rounds_sz, 0, offset(block), Mttkrp.cols());
        }
    }
} // end of namespace v2

namespace v2
{
    inline void mttkrp( const MatrixXd &X_mat, const MatrixXd &KR_p, const VectorXi Tns_dims, int Mode,  const unsigned int n_thrds, MatrixXd &Mttkrp)
    {
        Eigen :: setNbThreads(1);

        int rows_X_mat = X_mat.rows();
        int cols_X_mat = X_mat.cols();
    
        VectorXi rest_dims = Tns_dims;
        rest_dims(Mode) = 1;
        int max_dim = rest_dims.maxCoeff();
        int rounds;
        int offset;

        int cols_X_mat_full = rest_dims.prod();          // for the full

        Mttkrp.setZero();

        if( cols_X_mat < cols_X_mat_full)                //X_mat_sub
        {   
            offset = cols_X_mat/5;                      //trial
            rounds = 5;
        }
        else
        {
            offset = cols_X_mat_full/max_dim;
            rounds = max_dim;
        }
        
        #pragma omp parallel for reduction(sum: Mttkrp) default(shared) num_threads(n_thrds)
        for(int block = 0; block < rounds; block++ )
        {
            Mttkrp.noalias() += X_mat.block(0, block * offset, Mttkrp.rows(), offset) * KR_p.block(block * offset, 0, offset, Mttkrp.cols());
        }
    }
} // end of namespace v2

namespace partial // This is only for the cost function computations (USE_COST_FUN)
{
    // Calculates the full MTTKRP without computing the full Khatrirao
    // template <std::size_t TNS_ORDER>
    // inline void mttkrp(const MatrixXd &X_mat_full, const std::array<MatrixXd, TNS_ORDER> &Factors, const VectorXi tns_dims, int cur_mode, const unsigned int n_thrds, MatrixXd &MTTKRP_full)
    // {
        // Eigen :: setNbThreads(1);
        // int kr_dim = TNS_ORDER - 1;
        // MTTKRP_full.setZero();

        // // Here we create the kr_indices corresponding to factors participating in the kr product
        // VectorXi kr_factors(kr_dim);
        // for(int i = 0, j = 0; i < tns_dims, j < kr_dim; i++, j++)
        // {
        //     if(i == cur_mode)
        //     {   
        //         j--;
        //         continue;
        //     }
        //     kr_factors(j) = i;
        // }

        // int kr_rows = tns_dims.prod()/tns_dims(cur_mode);

        // int num_blocks = kr_rows/tns_dims(kr_factors(0));
        
        // // Create the block offset for each factor
        // VectorXi block_offset(kr_dim - 1);
        // int num_blocks_tmp = num_blocks;
        // for(int idx = kr_dim - 1; idx > 0; idx--)
        // {
        //     block_offset(idx - 1) = num_blocks_tmp/tns_dims(idx);
        //     num_blocks_tmp = block_offset(idx - 1);

        // }
    // }    
    template <std::size_t  TNS_ORDER>
    void mttpartialkrp(const int tensor_order, const Ref<const VectorXi> tensor_dims, const int tensor_rank, const int current_mode,
            std::array<MatrixXd, TNS_ORDER> &Init_Factors, const Ref<const MatrixXd> Tensor_X, MatrixXd &MTTKRP,
            const unsigned int num_threads)
    {
        #ifndef EIGEN_DONT_PARALLELIZE
            Eigen::setNbThreads(1);
        #endif

        MTTKRP.setZero();

        int mode_N = tensor_order - 1;

        int mode_1 = 0;

        if (current_mode == mode_N)
        {
            mode_N--;
        }
        else if (current_mode == mode_1)
        {
            mode_1 = 1;
        }

        // MatrixXd PartialKR(tensor_dims(mode_1), tensor_rank);

        // dim = I_(1) * ... * I_(current_mode-1) * I_(current_mode+1) * ... * I_(N)
        int dim = tensor_dims.prod() / tensor_dims(current_mode);

        // num_of_blocks = I_(mode_1+1) x I_(mode_1+2) x ... x I_(mode_N), where <I_(mode_1)> #rows of the first factor
        int num_of_blocks = dim / tensor_dims(mode_1);

        VectorXi rows_offset(tensor_order - 2);
        for (int ii = tensor_order - 3, jj = mode_N; ii >= 0; ii--, jj--)
        {
            if (jj == current_mode)
            {
                jj--;
            }
            if (ii == tensor_order - 3)
            {
                rows_offset(ii) = num_of_blocks / tensor_dims(jj);
            }
            else
            {
                rows_offset(ii) = rows_offset(ii + 1) / tensor_dims(jj);
            }
        }


        #pragma omp parallel for reduction(sum: MTTKRP) schedule(static, 1) num_threads(num_threads) proc_bind(close)
        for (int block_idx = 0; block_idx < num_of_blocks; block_idx++)
        {
            // Compute Kr = KhatriRao(A_(mode_N)(l,:), A_(mode_N-1)(k,:), ..., A_(2)(j,:))
            // Initiallize vector Kr as Kr = A_(mode_N)(l,:)
            MatrixXd Kr(1, tensor_rank);
            
            Kr = Init_Factors[mode_N].row((block_idx / rows_offset(tensor_order - 3)) % tensor_dims(mode_N));
            MatrixXd PartialKR(tensor_dims(mode_1), tensor_rank);

            // compute "partial" KhatriRao as a recursive Hadamard Product : Kr = Kr .* A_(jj)(...,:)
            for (int ii = tensor_order - 4, jj = mode_N - 1; ii >= 0; ii--, jj--)
            {
                if (jj == current_mode)
                {
                    jj--;
                }
                Kr = (Init_Factors[jj].row((block_idx / rows_offset(ii)) % tensor_dims(jj))).cwiseProduct(Kr);
            }

            // Compute block of KhatriRao, using "partial" vector Kr and first Factor A_(mode_1), as : KhatriRao(Kr, A_(mode_1)(:,:))
            for (int row = 0; row < tensor_dims(mode_1); row++)
            {
                PartialKR.row(row)  = ((Init_Factors[mode_1].row(row)).cwiseProduct(Kr));
            }
            
            MTTKRP.noalias() += Tensor_X.block(0, block_idx * tensor_dims(mode_1), tensor_dims(current_mode), tensor_dims(mode_1)) * PartialKR;
        }
        #ifndef EIGEN_DONT_PARALLELIZE
            Eigen::setNbThreads(num_threads);
        #endif
    }


        // int last_mode = TNS_ORDER - 1;
        // int first_mode = 0;                              
        // int kr_oper = TNS_ORDER - 2;

        // if(cur_mode == last_mode)
        // {
        //     last_mode--;
        // }
        // else if(cur_mode == first_mode)
        // {
        //     first_mode = 1;
        // }

        // int kr_rows = tns_dims.prod()/tns_dims(cur_mode);

        // int num_blocks = kr_rows/tns_dims(first_mode);

        // VectorXi block_offset(kr_oper);
        // for(int i = kr_oper - 1, int mode = last_mode; i >= 0; i--, mode--)
        // {
        //     if(mode = cur_mode)
        //     {
        //         continue;
        //     }
        //     if(i == kr_oper - 1)
        //     {
        //         block_offset(i) = num_blocks/tns_dims(mode)
        //     }
        //     else
        //     {
        //         block_offset(i) = block_offset(i+1)/tns_dims(mode)
        //     }
            
        // }

} // end  of namespace partial
    
#endif 
