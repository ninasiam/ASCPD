#include "../../include/master_library.hpp"
#include "../../include/sampling_funs.hpp"
#include "../../include/solve_BrasCPaccel.hpp"
#include "../../include/cpdgen.hpp"
#include <string>

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
        srand((unsigned) time(NULL));                 
    #endif

    const int TNS_ORDER = 3;                                      // Declarations
    const size_t R = 2;
    
    VectorXi tns_dims(TNS_ORDER);

    Eigen::Tensor< double, TNS_ORDER > True_Tensor;
    Eigen::Tensor< double, TNS_ORDER > Est_Tensor_from_factors; // with no dims
    Eigen::Tensor< double, TNS_ORDER > Diff_Tensor;

    std::array<MatrixXd, TNS_ORDER> Init_Factors;

    // Assign values
    tns_dims.setConstant(3); 

    //Initialize the true tensor
    True_Tensor.resize(tns_dims);
    True_Tensor.setRandom<Eigen::internal::UniformRandomGenerator<double>>();

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

    CpdGen( tns_dims, Init_Factors, R, Est_Tensor_from_factors);
    cout << "Est_Tensor_from_factors" << " = \n " << Est_Tensor_from_factors << endl;


    // Frobenius norm of Tensor
    Eigen::Tensor< double, 0 > frob_X  = Est_Tensor_from_factors.square().sum().sqrt();  
    cout << "Frob_X:"  << frob_X << endl; 

    // // Compute the matricization in terms of A
    MatrixXd KR_full1(tns_dims(2)*tns_dims(1), R);
    Khatri_Rao_Product( Init_Factors[2], Init_Factors[1], KR_full1);


    MatrixXd X_A_mat = Init_Factors[0]*KR_full1.transpose();

    cout << "X_A_mat   " << " = \n " << X_A_mat << endl;

    // True tensor - init_tensor
    Diff_Tensor = True_Tensor - Est_Tensor_from_factors;
    
    Eigen::Tensor< double, 0 > f_val_init  = Diff_Tensor.square().sum().sqrt();  
    cout << "f_val_init:"  << f_val_init << endl; 

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

    CpdGen( tns_dims, Init_Factors, R, Est_Tensor_from_factors);
    cout << "Est_Tensor_from_factors 2" << " = \n " << Est_Tensor_from_factors << endl;

    // Frobenius norm of Tensor
    frob_X  = Est_Tensor_from_factors.square().sum().sqrt();  
    cout << "Frob_X 2:"  << frob_X << endl; 

    return 0;
    
}