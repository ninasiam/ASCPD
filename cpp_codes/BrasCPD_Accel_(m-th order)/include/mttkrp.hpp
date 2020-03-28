#ifndef MTTKRP_HPP
#define MTTKRP_HPP

#include <omp.h>
#include "master_library.hpp"
#include "omp_lib.hpp"

/* IN : matricization of tensor, Khatri-rao Product, tns_dims, current mode, number of threads
   OUT: Matricized tensor times Khatri - rao */

inline void mttkrp( const MatrixXd &X_mat, const MatrixXd &KR_p, const VectorXi Tns_dims, int Mode,  const unsigned int n_thrds, MatrixXd &Mttkrp)
{
    Eigen :: setNbThreads(1);

    int rows_X_mat = X_mat.rows();
    int cols_X_mat = X_mat.cols();
   
    VectorXi rest_dims = Tns_dims;
    rest_dims(Mode) = 1;
    int max_dim = rest_dims.maxCoeff();
    int rounds;
    VectorXi offset;

    int cols_X_mat_full = rest_dims.prod();          // for the full

    Mttkrp.setZero();

    if( cols_X_mat < cols_X_mat_full)                //if X_mat_sub has fewer columns (e.g bs(mode))
    {   
        rounds = n_thrds;                               // Number of blocks 
        rounds_sz = cols_X_mat / n_thrds;               // Quot
        residual = cols_X_mat % n_thrds;                // Residual, in case the cols of X / n_thrds leaves a residual 
        if( residual != 0)                              // in case we have a residual
        {
            VectorXi offset_tmp(rounds,1);
            offset_tmp.setConstant(rounds_sz);          // create offset vector with rounds_sz offset for each block
            offset_tmp(rounds - 1) = rounds_sz + residual; //in the last one add the residual
            offset = offset_tmp;
        }
        else
        {
            offset.setConstant(rounds_sz);                      
            
        }
        

    }
    else //for the full case
    {   
        rounds = max_dim;
        VectorXi offset_tmp(rounds,1);
        offset_tmp.setConstant(cols_X_mat_full/max_dim);
        offset = offset_tmp;          //It would never enter when we have blocksize, for the full case
        
    }
    
    #pragma omp parallel for reduction(sum: Mttkrp) default(shared) num_threads(n_thrds)
    for(int block = 0; block < rounds; block++ )
    {
        Mttkrp.noalias() += X_mat.block(0, block * offset(block), Mttkrp.rows(), offset(block)) * KR_p.block(block * offset(block), 0, offset(block), Mttkrp.cols());
    }
}


#endif 
