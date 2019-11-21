#ifndef SAMPLING_FUNS_HPP
#define SAMPLING_FUNS_HPP

#include "master_library.hpp"
#include "khatri_rao_prod.hpp"

inline void Sampling_Operator(int order, VectorXi block_size, VectorXi dims, VectorXi &F_n, int &factor)
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

void Sampling_Sub_Matrices(const VectorXi &F_n, MatrixXd &KhatriRao, const MatrixXd &X, const MatrixXd &U1, const MatrixXd &U2,  MatrixXd &KhatriRao_sub, MatrixXd &X_sub)
{   
    int R = KhatriRao.cols();
    int J_n = KhatriRao.rows();
    int bz = F_n.size();
    // cout << bz << endl;
    MatrixXd KhatriRao_T(R, J_n);

    Khatri_Rao_Product(U1, U2, KhatriRao);                          // Compute the full Khatri-Rao Product
    // cout << KhatriRao << endl;
    KhatriRao_T = KhatriRao.transpose();

    for(int col_H = 0; col_H < bz; col_H++)
    {
        KhatriRao_sub.col(col_H) = KhatriRao_T.col(F_n(col_H));     //Create KhatriRao_sub (transpose)
    }
    
    for(int col_X = 0; col_X < bz; col_X++)
    {
        X_sub.col(col_X) = X.col(F_n(col_X));
    }
    // cout << "Mpike 1" << endl;
}
#endif