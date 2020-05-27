#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/calc_gradient.hpp"

using namespace std::chrono_literals;

int main(int argc, char **argv){

    int bs = 200;
    int R  = 40;
    double L, mu;
    double L_eig, mu_eig;

    nanoseconds stop_t_svd = 0ns;
    nanoseconds stop_t_eig = 0ns;

    MatrixXd A = (MatrixXd::Random(bs, R) + MatrixXd::Ones(bs, R))/2;     // Matrix with analogous dimensions with KR_Prod in our problem
    MatrixXd A_T_A = A.transpose()*A;

    auto t1_svd  = high_resolution_clock::now();
    svd::Compute_mu_L(L, mu, A);
    auto t2_svd = high_resolution_clock::now();
    stop_t_svd = duration_cast<nanoseconds>(t2_svd - t1_svd);


    auto t1_eig  = high_resolution_clock::now();
    eig::Compute_mu_L(L_eig, mu_eig, A_T_A);
    auto t2_eig  = high_resolution_clock::now();
    stop_t_eig = duration_cast<nanoseconds>(t2_eig - t1_eig);


    cout << "Time  SVD = " << stop_t_svd.count() * (1e-9) << "s" << endl;
    cout << "Time  EIG = " << stop_t_eig.count() * (1e-9) << "s" << endl;

    cout << "L(svd) = " << " " << L << " L(eig) = " <<  " " << sqrt(L_eig) << endl; 
    cout << "mu(svd) = " << " " << mu << " mu(eig) = " <<  " " << sqrt(mu_eig) << endl; 


}