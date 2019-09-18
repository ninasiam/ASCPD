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

    std::cout << "ERM via katyusha method for the averages least squares cost function" << endl;

    int n, d, S, m, k, outer_iter, MAX_OUTER_ITER;
    double L, sigma, f_value, f_star, tau_1, alpha;
    float tau_2, epsilon;
  
    n = 2000;
    d = 500;
    MAX_OUTER_ITER = 500000;
    S = 100;

    MatrixXd A(n,d);
    MatrixXd A_t(d,n);
    MatrixXd A_t_A(n,n);
    VectorXd b(n,1);

    VectorXd x_init(d,1);
    VectorXd full_grad(d,1);
    VectorXd stoch_grad(d,1);
    VectorXd x_star(d,1);
    VectorXd x_hat(d,1), x_next(d,1);
    VectorXd y(d,1), sum_y(d,1), y_next(d,1);
    VectorXd z(d,1), z_next(d,1);


    
    // Algorithm Initializations

    A.setRandom(n,d);
    b.setRandom(n,1);

    x_init.setRandom(d,1);
 

    // cout << "A matrix is:" << A << endl;
    // cout << "b vector is:" << b << endl;

    A_t = A.transpose();
    A_t_A.noalias() = A_t*A;

    // cout << "A_t_A matrix is:" << A_t_A << endl;

    // x_star = inv((A'*A))*A'*b;
    // f_star = (1/(2*n))*norm(A*x_star - b)^2;

    x_star = A_t_A.inverse()*A_t*b;
    Calculate_fval(A, b, x_star, &f_value);
    f_star = f_value;

    std::cout << "f_star = " << f_star << endl;

    Compute_parameters(A_t_A, &L, &sigma);
  
    std::cout << "Parameters: L = "<< L << ",sigma = " << sigma << endl;
    std::cout << "Dimensions: n = "<< n << ",d = " << d << endl;

    Calculate_fval(A, b, x_init, &f_value);


    x_hat = x_init;
    y = x_init;
    z = x_init;
    sum_y = y;

    m = 2*n;
    tau_2 = 0.5;
    tau_1 = min(sqrt(m*sigma)/sqrt(3*L),0.5);
    alpha = 1/(3*tau_1*L);

    outer_iter = 0;
    epsilon = 10^(-4);

    
    std::cout << "-----> Katyusha outer loop <-----" << endl;
    std::cout << "f_value init = " << f_value << endl;

    std::cout << "Outer Iter     ------    f_value" << endl;

    while(1){

        full_grad = Compute_gradient(A_t, A_t_A, b, x_hat, d, n);

        for(int j = 0; j < m; j++){

            k = outer_iter*m + j;
            x_next = tau_1*z + tau_2*x_hat + (1 - tau_1 - tau_2)*y;


            int rand_int = Random_int_generator(n);
            // cout << "rand int" << rand_int << endl;

            stoch_grad = full_grad + (A.row(rand_int)*x_next - b(rand_int))*A_t.col(rand_int) + (A.row(rand_int)*x_hat - b(rand_int))*A_t.col(rand_int);

            z_next = z - alpha*stoch_grad;

            y_next = x_next - (1/(3*L))*stoch_grad;

            // sum_y.noalias() = sum_y + y;

            z = z_next;
            y = y_next;
        }

        x_hat = y;
        

        Calculate_fval(A, b, x_hat, &f_value);

        outer_iter++;
        std::cout << "  "<<outer_iter << "\t\t\t " << f_value << endl;

        if(abs(f_value - f_star) < epsilon || outer_iter*m > MAX_OUTER_ITER ){

            std::cout << "Stopping Criterion has been met" << endl;
            std::cout << "measure:" << abs(f_value - f_star) << endl;
            break;
        }
        
    }

    return 0;


}