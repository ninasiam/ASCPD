#ifndef MTTKRP_HPP
#define MTTKRP_HPP

#include <omp.h>
#include "master_library.hpp"
#include "omp_lib.hpp"

/* IN : matricization of tensor, Khatri-rao Product, tns_dims, current mode
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


#endif 
