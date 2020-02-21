#include "../../include/master_lib.hpp"
#include "../../include/MatrixFISTA.hpp"

#include <string>

int main(int argc, char **argv)
{
    int N = 10; 
    int M = 10;
    int R = 2;

    MatrixXd MTTKRP = MatrixXd::Random(N,R);
    MatrixXd V = MatrixXd::Random(R, R);
    V = V.transpose()*V;
    MatrixXd Factor = MatrixXd::Random(N, R);
    
    double lambda = 0.5;

    std::cout << "Initial Factor = " << Factor << std::endl;

    MatrixFISTA(MTTKRP, V, Factor, lambda);
    
    std::cout << "New Factor = " << Factor << std::endl;

}