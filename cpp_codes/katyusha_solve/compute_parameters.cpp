#include "kat_library.h"

using namespace Eigen;

void Compute_parameters(const MatrixXd &A_t_A, double *L, double *sigma){
    
    JacobiSVD<MatrixXd> svd(A_t_A, ComputeThinU | ComputeThinV);
    *L = svd.singularValues().maxCoeff();
    *sigma = svd.singularValues().minCoeff();
}