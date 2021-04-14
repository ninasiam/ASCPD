#include "../../include/master_library.hpp"
#include "../../include/solve_BRAS_NN.hpp"


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

	VectorXi dims(3,1);
	int order = 3;											// Tensor Order
	int I, J, K;										    // Tensor Dimensions
	int R;												    // Rank of factorization
	// time_t start_t, stop_t;							    // Timer
	// start_t = time(0);
	// srand(time(NULL));

	const double AO_tol = 1e-4;							    // Tolerance for AO Algorithm
	int max_iter_mttkrp = 20;								// Maximum Number of iterations
	int AO_iter  = 1;										// Iterations counter

	VectorXi block_size(3,1);								// |
	block_size.setConstant(30);								// | Parameters for stochastic
															// |
	VectorXi F_n(block_size(0),1);							// | fibers to be selected
	int factor;												// |
	// VectorXi kr_idx(2,1);								// |


	double frob_X;				                		    // Frobenius norm of Tensor
	double f_value;										    // Objective Value

    // Set_Info(&R, &I, &J, &K, "Data_cpp/info.bin");			// Initialize tensor size and rank from file

	R = 10, I = 100, J = 100, K = 100;
    cout << "R=" << R << ", I=" << I << ", J=" << J << ", K=" << K << endl;
	cout << "block size=" << block_size.transpose() << endl;
	dims(0) = I, dims(1) = J, dims(2) = K;
	//------------------------------> Matrix Initializations <-------------------------------------

	MatrixXd A(I,R), B(J,R), C(K,R);						// Factors A,B,C
	MatrixXd A_init(I,R), B_init(J,R), C_init(K,R);
	MatrixXd A_T_A(R,R), B_T_B(R,R), C_T_C(R,R); 

	MatrixXd X_A(I, size_t(K * J));							// |
	MatrixXd X_B(J, size_t(K * I));							// | Tensor Matricization 
	MatrixXd X_C(K, size_t(I * J));							// |

	MatrixXd KhatriRao_CB(size_t(J * K), R);				// |
	MatrixXd KhatriRao_CA(size_t(I * K), R);				// | Khatri Rao products
	MatrixXd KhatriRao_BA(size_t(I * J), R);				// |

	// Read_Data(A, B, C, X_A, X_B, X_C, I, J, K, R);		// Read Factors and matricization from file

	//Write factors to files
	A = (MatrixXd::Random(I, R) + MatrixXd::Ones(I ,R))/2;	
	B = (MatrixXd::Random(J, R) + MatrixXd::Ones(J ,R))/2;	
	C = (MatrixXd::Random(K, R) + MatrixXd::Ones(K ,R))/2;	

	// cout << "A = " << A << endl;
	// cout << "B = " << B << endl;
	// cout << "C = " << C << endl;
	MatrixXd A_T = A.transpose();
	Write_to_File(I, R, A_T, "../Data_cpp/A.bin");
	MatrixXd B_T = B.transpose();	
	Write_to_File(J, R, B_T, "../Data_cpp/B.bin");
	MatrixXd C_T = C.transpose();	
	Write_to_File(K, R, C_T, "../Data_cpp/C.bin");


	A_init = (MatrixXd::Random(I, R) + MatrixXd::Ones(I ,R))/2;	
	B_init = (MatrixXd::Random(J, R) + MatrixXd::Ones(J ,R))/2;	
	C_init = (MatrixXd::Random(K, R) + MatrixXd::Ones(K ,R))/2;	

	// cout << "A_init = " << A_init << endl;
	// cout << "B_init = " << B_init << endl;
	// cout << "C_init = " << C_init << endl;

	MatrixXd A_init_T = A_init.transpose();
	Write_to_File(I, R, A_init_T, "../Data_cpp/A_init.bin");
	MatrixXd B_init_T = B_init.transpose();	
	Write_to_File(J, R, B_init_T, "../Data_cpp/B_init.bin");
	MatrixXd C_init_T = C_init.transpose();	
	Write_to_File(K, R, C_init_T, "../Data_cpp/C_init.bin");

	Khatri_Rao_Product(C, B, KhatriRao_CB);
	Khatri_Rao_Product(C, A, KhatriRao_CA);
	Khatri_Rao_Product(B, A, KhatriRao_BA);

	X_A = A * KhatriRao_CB.transpose();
	X_B = B * KhatriRao_CA.transpose();
	X_C = C * KhatriRao_BA.transpose();
	//----------------------------> F_value Computation <------------------------------------------
	
	Solve_brasNN(A_init, B_init, C_init, X_A, X_B, X_C, KhatriRao_CB, KhatriRao_CA, KhatriRao_BA,  I, J, K, R, block_size, AO_tol, max_iter_mttkrp, f_value, AO_iter);

	
}