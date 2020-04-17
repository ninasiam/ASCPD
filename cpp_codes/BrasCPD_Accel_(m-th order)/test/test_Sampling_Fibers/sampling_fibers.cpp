#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"


#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL)+std::rand());                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const int R = 2;
    VectorXi tns_dims(TNS_ORDER,1), block_size(TNS_ORDER,1);
    Eigen::Tensor< double, TNS_ORDER > True_Tensor;

    // Assign values
    tns_dims.setConstant(10); 
    block_size.setConstant(5);

    //Initialize the tensor
    True_Tensor.resize(tns_dims);
    True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();           // Tensor using UniformRandomGenerator
    
    double* Tensor_pointer = True_Tensor.data();


    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;

    int mode = 2;
    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXi factor_idxs(block_size(mode),TNS_ORDER-1);
    
    MatrixXd T_mode(tns_dims(mode), block_size(mode));
    

    symmetric::Sample_Fibers(Tensor_pointer,  tns_dims,  block_size,  mode,
                      idxs, factor_idxs, T_mode);

    cout << "Matricization mode 3 = \n " << T_mode << endl;

    return 0;

}