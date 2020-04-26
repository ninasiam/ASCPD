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
        Eigen :: setNbThreads(1);

        int rows_X_mat = X_mat.rows();
        int cols_X_mat = X_mat.cols();
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
}

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
}

#endif 
