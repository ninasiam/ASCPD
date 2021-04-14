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
    tns_dims.setConstant(3); 
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

    int mode = 1;
    MatrixXi idxs(block_size(mode),TNS_ORDER);
    MatrixXd T_mode(tns_dims(mode), block_size(mode));
    MatrixXd KR_sampled(tns_dims(1)*tns_dims(2), R);
    MatrixXd KR_Full(tns_dims(1)*tns_dims(2), R);

    cout << "Test 0: " << endl; 
    std::vector<std::vector<int>> fibers_idxs =  sorted::Sample_fibers<TNS_ORDER>(Tensor_pointer, tns_dims, block_size, mode,
                    idxs, T_mode);
    // Number of rows; 
    int m = fibers_idxs.size();  
    int n = fibers_idxs[0].size(); 

    cout << "idxs (Full idxs list before sorting): " << "\n" << idxs << endl;
    // Displaying the 2D vector after sorting 
    cout << "The Matrix after sorting 1st row is:\n"; 
    for (int i=0; i<m; i++) 
    { 
        for (int j=0; j<n ;j++) 
            cout << fibers_idxs[i][j] << " "; 
        cout << endl; 
    } 
    cout << "Matricization mode ="<< mode << " \n " << T_mode << endl;

    // sampling KR-product
    sorted::Sample_KhatriRao(mode, R, fibers_idxs, True_Factors, KR_sampled);
    cout << "KR sampled = \n " << KR_sampled << endl;

    Khatri_Rao_Product(True_Factors[2], True_Factors[0], KR_Full);
    cout << "KR FULL = \n " << KR_Full << endl;

    return 0;

}