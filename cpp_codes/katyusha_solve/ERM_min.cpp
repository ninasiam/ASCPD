/*----------------------------------------------------------------------------------*/
/*                               ERM via katyusha                                   */
/*                                                                                  */
/*Ioanna Siaminou                                                                   */
/*9/9/2019                                                                          */
/*----------------------------------------------------------------------------------*/

#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include "kat_library.h"
#include <limits>
#include <iostream>
#include <cstdlib>

// Run locally
// #include <Eigen/Dense>
// #include <Eigen/Core>

// Run on Dali
#include "/home/ninasiam/Desktop/PARTENSOR/Libraries/eigen3/Eigen/Dense"
#include "/home/ninasiam/Desktop/PARTENSOR/Libraries/eigen3/Eigen/Core"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){

    cout << "ERM via katyusha method for the averages least squares cost function" << endl;

    int n, d, S, m, k, outer_iter, MAX_OUTER_ITER;
    double L, sigma, f_value, tau_1, alpha;
    float tau_2, epsilon;
  
    MatrixXd A(n,d);
    MatrixXd A_t(d,n);
    MatrixXd A_t_A(n,n);
    VectorXd b(n,1);

    VectorXd x_init(d,1);
    VectorXd full_grad(d,1);
    VectorXd stoch_grad(d,1);
    VectorXd x(d,1);
    VectorXd x_hat(d,1), x_next(d,1);
    VectorXd y(d,1), sum_y;
    VectorXd z(d,1), z_next(d,1);

    n = 50;
    d = 5;
    MAX_OUTER_ITER = 200;
    S = 100;
    
    // Algorithm Initializations

    A.setRandom(n,d);
    b.setRandom(n,1);

    x_init.setZero(d,1);
    sum_y.setZero(d,1);

    // cout << "A matrix is:" << A << endl;
    // cout << "b vector is:" << b << endl;

    A_t = A.transpose();
    A_t_A.noalias() = A.transpose()*A;

    // cout << "A_t_A matrix is:" << A_t_A << endl;

    
    Compute_parameters(A_t_A, &L, &sigma);

    cout << "Parameters: L = "<< L << ",sigma = " << sigma << endl;
    cout << "Dimensions: n = "<< n << ",d = " << d << endl;

    Calculate_fval(A, b, x_init, &f_value);

    x = x_init;
    x_hat = x_init;
    y = x_init;
    z = x_init;

    tau_2 = 0.5;
    tau_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
    alpha = 1/(3*tau_1*L);
    m = 2*n;
    outer_iter = 0;
    epsilon = 10^(-4);

    cout << "f_value init = " << f_value << endl;
    
    cout << "-----> Katyusha outer loop <-----" << endl;
    cout << "f_value init = " << f_value << endl;


    while(1){

        full_grad = Compute_gradient(A_t, A_t_A, b, x_hat, d, n);

        for(int j = 0; j < m; j++){

            k = outer_iter*m + j;
            x_next = tau_1*z + tau_2*x_hat + (1 - tau_1 - tau_2)*y;

            int rand_int = Random_int_generator(n);

            stoch_grad = full_grad + (A.row(rand_int)*x - b(rand_int))*A.row(rand_int).transpose() + (A.row(rand_int)*x_hat - b(rand_int))*A.row(rand_int).transpose();

            z_next = z - alpha*stoch_grad;

            y = x_next - (1/(3*L))*stoch_grad;

            sum_y = sum_y + y;

            z = z_next;
        }

        x_hat = y;
        outer_iter++;

        Calculate_fval(A, b, x_hat, &f_value);
       
        cout << "f_value init = " << f_value << endl;

        if(f_value < epsilon || outer_iter > MAX_OUTER_ITER){

            cout << "Stopping Criterion has been met" << endl;
            break;
        }
    }

    return 0;


}