#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/cpdgen.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on, the data will be different for every run (including the random fibers)
                                                                    
int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL));                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const int R = 2;
    VectorXi tns_dims(TNS_ORDER,1), block_size(TNS_ORDER,1);
    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> True_Factors;

    // Assign values
    tns_dims.setConstant(4); 
    block_size.setConstant(3);
    ////////////////////////////////////////////////////////////////////////////////////////

    // Initialize the tensor using tensor module
    // True_Tensor.resize(tns_dims);
    // True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();           // Tensor using UniformRandomGenerator
    

    // Using true factors
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        True_Factors[factor] = (MatrixXd::Random(tns_dims(factor), R) + MatrixXd::Ones(tns_dims(factor), R))/2;

    }
    CpdGen( tns_dims, True_Factors, R, True_Tensor);
    ////////////////////////////////////////////////////////////////////////////////////////

    double* Tensor_pointer = True_Tensor.data();

    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;
    cout << "True Tensor: " << "\n" << True_Tensor << endl;

    int mode = 0;
    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    
    MatrixXd T_mode(tns_dims(mode), block_size(mode));
    
    cout << "Test 1: " << endl; 
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                      idxs, factor_idxs, T_mode);

    cout << "idxs: " << "\n" << idxs << endl;
    cout << "Matricization mode 0 = \n " << T_mode << endl;

    mode = 1;
    cout << "Test 2: " << endl;
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                      idxs, factor_idxs, T_mode);

    cout << "idxs: " << "\n" << idxs << endl;
    cout << "Matricization mode 1 = \n " << T_mode << endl;


    return 0;

}