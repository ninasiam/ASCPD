#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL)+std::rand());                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const int R = 2;
    
    VectorXi tns_dims(TNS_ORDER);
    VectorXi block_size(TNS_ORDER);

    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;

    // Assign values
    tns_dims.setConstant(10); 
    block_size.setConstant(5);

    //Initialize the tensor
    True_Tensor.resize(tns_dims);
    True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();           // Tensor using UniformRandomGenerator
    
    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;

    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = True_Tensor.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 

    cout << "True Tensor: \n" <<True_Tensor << endl;

    double* Tensor_pointer = True_Tensor.data();

    // Create Init Factors
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = MatrixXd::Random(tns_dims(factor), R);
        cout << "Init_factor: " << Init_Factors[factor] << endl;
    }

    int mode = 0;
    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    MatrixXd T_mode(tns_dims(mode), block_size(mode));            // Matricization Sampled

    // Select the fibres and form the matricization
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                             idxs, factor_idxs, T_mode);

    // Form the sampled Khatri-Rao
    MatrixXd KR_sampled(block_size(mode), R);                     // Khatri-Rao Sampled
    MatrixXd KR_full(tns_dims(2)*tns_dims(1), R);                 // Khatri-Rao full (for testing purposes) !!! Change if the mode is different !!!
    symmetric::Sample_KhatriRao( mode, R, idxs, Init_Factors, KR_sampled);

    cout << "KR_sampled   " << " = \n " << KR_sampled << endl;

    Khatri_Rao_Product( Init_Factors[2], Init_Factors[1], KR_full);
    cout << "KR_full   " << " = \n " << KR_full << endl;

    // double AO_tol = 0.001;
    // int MAX_MTTKRP = 10;
    
    // symmetric::solve_BrasCPaccel(AO_tol, MAX_MTTKRP, R, frob_X, tns_dims, block_size, Init_Factors, Tensor_pointer);
    return 0;
}