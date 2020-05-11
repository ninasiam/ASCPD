#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL));                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const int R = 2;
    
    VectorXi tns_dims(TNS_ORDER);
    VectorXi block_size(TNS_ORDER);

    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;
    std::array<MatrixXd, TNS_ORDER> True_Factors;


    // Assign values
    tns_dims.setConstant(4); 
    tns_dims(1) = 3;
    block_size.setConstant(5);

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
    
    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;

    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = True_Tensor.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 
    cout << "True Tensor" << True_Tensor << endl;
    double* Tensor_pointer = True_Tensor.data();

    // Create Init Factors
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = MatrixXd::Random(tns_dims(factor), R);
        // cout << "Init_factor: " << Init_Factors[factor] << endl;
    }

    int mode = 1;
    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    MatrixXd T_mode(tns_dims(mode), block_size(mode));            // Matricization Sampled

    // Select the fibres and form the matricization
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                             idxs, factor_idxs, T_mode);
    cout << "idxs" << idxs << endl;
    cout << "factor_idxs" << factor_idxs << endl;

    cout << "T_mode" << T_mode << endl;


    // Form the sampled Khatri-Rao
    MatrixXd KR_sampled(block_size(mode), R);                     // Khatri-Rao Sampled
    MatrixXd KR_full(tns_dims(2)*tns_dims(0), R);                 // Khatri-Rao full (for testing purposes) !!! Change if the mode is different !!!
    symmetric::Sample_KhatriRao( mode, R, idxs, Init_Factors, KR_sampled);

    cout << "KR_sampled   " << " = \n " << KR_sampled << endl;

    Khatri_Rao_Product( Init_Factors[2], Init_Factors[0], KR_full);
    cout << "KR_full   " << " = \n " << KR_full << endl;

    return 0;
}