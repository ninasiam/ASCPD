#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL)+std::rand());                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const int R = 10;
    
    VectorXi tns_dims(TNS_ORDER);
    VectorXi block_size(TNS_ORDER);

    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;

    // Assign values
    tns_dims.setConstant(100); 
    block_size.setConstant(20);

    //Initialize the tensor
    True_Tensor.resize(tns_dims);
    True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();           // Tensor using UniformRandomGenerator
    
    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;

    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = True_Tensor.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 

    double* Tensor_pointer = True_Tensor.data();

    // Create Init Factors
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = MatrixXd::Random(tns_dims(factor), R);
    }


    double AO_tol = 0.001;
    int MAX_MTTKRP = 10;
    
    symmetric::solve_BrasCPaccel(AO_tol, MAX_MTTKRP, R, frob_X, tns_dims, block_size, Init_Factors, Tensor_pointer);
    return 0;
}