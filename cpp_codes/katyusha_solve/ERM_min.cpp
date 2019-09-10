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

#include <Eigen/Dense>
#include <Eigen/Core>
#include <cstdlib>
using namespace std;
using namespace Eigen;

int main(int argc, char **argv){

    cout << "ERM via katyusha method for the averages least squares cost function" << endl;

    int n, d, S;
    double L, sigma, f_value;
    n = 50;
    d = 5;
    S = 100;
    
    MatrixXd A(n,d);
    MatrixXd A_t_A(n,n);
    VectorXd b(n,1);

    VectorXd x_init(d,1);

    A.setRandom(n,d);
    b.setRandom(n,1);

    x_init.setZero(d,1);

    // cout << "A matrix is:" << A << endl;
    // cout << "b vector is:" << b << endl;

    A_t_A.noalias() = A.transpose()*A;

    // cout << "A_t_A matrix is:" << A_t_A << endl;

    Compute_parameters(A_t_A, &L, &sigma);

    cout << "Parameters: L = "<< L << ",sigma = " << sigma << endl;
    cout << "Dimensions: n = "<< n << ",d = " << d << endl;

    Calculate_fval(A, b, x_init, &f_value);

    cout << "f_value init = " << f_value << endl;
    
    cout << "-----> Katyusha outer loop <-----" << endl;


    return 0;


}