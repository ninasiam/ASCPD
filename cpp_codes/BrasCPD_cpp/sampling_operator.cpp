#include <iostream>
#include <random>
#include <cstdlib>
#include <bits/stdc++.h> 
#include "brascpd_functions.h"

using namespace std;
using namespace Eigen;

void Sampling_Operator(int order, VectorXi block_size, VectorXi dims, VectorXi &F_n, int &factor)
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