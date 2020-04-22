#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"
#include "../../include/cpdgen.hpp"
#include <string>

#define INITIALIZED_SEED 1                                        // if initialized seed is on the data will be different for every run (including the random fibers)

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
        srand((unsigned) time(NULL)+std::rand());                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const size_t R = 5;
    
    VectorXi tns_dims(TNS_ORDER);

    Eigen::Tensor< double, TNS_ORDER > True_Tensor_from_factors; // with no dims
    std::array<MatrixXd, TNS_ORDER> Init_Factors;

    // Assign values
    tns_dims.setConstant(20); 


    cout << "Tensor of order: " << TNS_ORDER << "\t ~Dimensions: " << tns_dims.transpose() << "\t ~Rank: "<< R << endl;

    // Create Init Factors, Write factors to Matlab to compare the results
    for(size_t factor = 0; factor < TNS_ORDER; factor++ )
    {
        Init_Factors[factor] = MatrixXd::Random(tns_dims(factor), R);
        std::string file_name1 = "../Data_cpp/initial_factor_";
        std::string file_subfix1 = std::to_string(factor);
        std::string file_extension1 = ".bin";
        file_name1 = file_name1 + file_subfix1 + file_extension1;

        MatrixXd Factor_to_write = Init_Factors[factor].transpose();
        Write_to_File(tns_dims(factor), R, Factor_to_write, file_name1.c_str());
    }

    CpdGen( tns_dims, Init_Factors, R, True_Tensor_from_factors);

    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = True_Tensor_from_factors.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 

    
    return 0;
    
}