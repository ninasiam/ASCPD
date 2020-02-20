#include "../../include/master_lib.hpp"
#include "../../include/MatrixFISTA.hpp"

#include <string>

int main(int argc, char **argv)
{
    int N = 100; 
    int M = 100;
    int R = 5;

    MatrixXd MTTKRP = MatrixXd::Random(N,R);
    MatrixXd V = MatrixXd::Random(R, R);
    MatrixXd Factor = MatrixXd::Random(N, R);
    
    double lambda = 0.5;
    
    double error = 1;

    std::cout << "Initial Factor = " << Factor << std::endl;

    MatrixFISTA(MTTKRP, V, Factor, lambda);
    
    std::cout << "New Factor = " << Factor << std::endl;

}