#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"
#include "../../include/cpdgen.hpp"

#define INITIALIZED_SEED 0                                        // if initialized seed is on the data will be different for every run (including the random fibers)

void Write_to_File(int nrows, int ncols, Ref<MatrixXd> Mat, const char *file_name){
	ofstream my_file(file_name, ios::out | ios::binary | ios::trunc);
	if (my_file.is_open()){
		my_file.write((char *) Mat.data(), nrows*ncols*sizeof(double));
		my_file.close();
	}
	else
		cout << "Unable to open file \n";
}

int main(int argc, char **argv){

    #if INITIALIZED_SEED                                          // Initialize the seed for different data
        srand((unsigned) time(NULL)                 
    #endif

    const int TNS_ORDER = 4;                                      // Declarations
    const int R = 20;
    
    VectorXi tns_dims(TNS_ORDER);
    VectorXi block_size(TNS_ORDER);

    Eigen::Tensor< double, 0 > f_value;
    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    Eigen::Tensor< double, TNS_ORDER > Init_Tensor;
    std::array<MatrixXd, TNS_ORDER> Init_Factors;
    std::array<MatrixXd, TNS_ORDER> True_Factors;

    // Assign values
    tns_dims.setConstant(100); 
    tns_dims(3) = 50;
    block_size.setConstant(280);

    //Initialize the tensor
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        True_Factors[factor] = (MatrixXd::Random(tns_dims(factor), R) + MatrixXd::Ones(tns_dims(factor), R))/2;

    }
    CpdGen( tns_dims, True_Factors, R, True_Tensor);

    //////////////////////////////////////////////////////////////

    // Initialize tensor using tensor module

    // True_Tensor.resize(tns_dims);
    // True_Tensor.setConstant(3/2);

    // True_Tensor = True_Tensor + True_Tensor.random();

    ///////////////////////////////////////////////////////////////

    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;
    cout << "Sampling of each mode with blocksize: " << block_size.transpose() << endl;
    // cout << True_Tensor << endl;

    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = True_Tensor.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 

    double* Tensor_pointer = True_Tensor.data();


    // Create Init Factors, Write factors to Matlab to compare the results
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = (MatrixXd::Random(tns_dims(factor), R) + MatrixXd::Ones(tns_dims(factor), R))/2;
        std::string file_name1 = "../Data_cpp/initial_factor_";
        std::string file_subfix1 = std::to_string(factor);
        std::string file_extension1 = ".bin";
        file_name1 = file_name1 + file_subfix1 + file_extension1;

        MatrixXd Factor_to_write = Init_Factors[factor].transpose();
        Write_to_File(tns_dims(factor), R, Factor_to_write, file_name1.c_str());
    }
    CpdGen( tns_dims, Init_Factors, R, Init_Tensor);
    f_value = (True_Tensor - Init_Tensor).square().sum().sqrt(); 

    VectorXi zero_vector(TNS_ORDER);
    zero_vector.setZero();
    Init_Tensor.resize(zero_vector);

    double AO_tol = 0.001;
    int MAX_MTTKRP = 40;
    
    symmetric::solve_BrasCPaccel(AO_tol, MAX_MTTKRP, R, frob_X, f_value, tns_dims, block_size, Init_Factors, Tensor_pointer, True_Tensor);
    return 0;
}