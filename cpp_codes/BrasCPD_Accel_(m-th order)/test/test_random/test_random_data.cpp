#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

int main(int argc, char **argv)
{   

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL));                 
    #endif

    const int TNS_ORDER = 3;                                      //Declarations
    const int R = 2;

    int mode;  
    VectorXi tns_dims(TNS_ORDER);
    VectorXi block_size(TNS_ORDER);
    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;

    // Assign values
    tns_dims.setConstant(10);
    block_size.setConstant(5);

    // Initialize the tensor
    True_Tensor.resize(tns_dims);
    True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();           // Tensor using UniformRandomGenerator

    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;

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

    // Check if the randomization is valid in mode selection
    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    symmetric::Sample_mode(TNS_ORDER, mode);
    cout << "Mode  = " << mode <<"\t\t\t"<< endl;

    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    MatrixXd T_mode(tns_dims(mode), block_size(mode));
    // Check if the idxs matrix is random
    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                             idxs, factor_idxs, T_mode);

    cout << "idxs  = " << idxs <<"\t\t\t"<< endl;

}