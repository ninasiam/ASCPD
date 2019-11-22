#include "../include/master_library.hpp"
#include "../include/solve_BRAS_NN.hpp"

int main(int argc, char **argv){

	VectorXi dims(3,1);
	int order = 3;											// Tensor Order
	int I, J, K;										    // Tensor Dimensions
	int R;												    // Rank of factorization
	// time_t start_t, stop_t;							    // Timer
	// start_t = time(0);
	srand(time(NULL));

	const double AO_tol = 1e-3;							    // Tolerance for AO Algorithm
	int max_iter = 5000;									// Maximum Number of iterations
	int AO_iter  = 1;										// Iterations counter

	VectorXi block_size(3,1);								// |
	block_size.setConstant(100);							// | Parameters for stochastic
	const double alpha_init = 1.2;							// | gradient
	const double beta = 1e-4;								// | 
	double alpha;											// |
	VectorXi F_n(block_size(0),1);							// | fibers to be selected
	int factor;												// |
	// VectorXi kr_idx(2,1);								// |


	double frob_X;				                		    // Frobenius norm of Tensor
	double f_value;										    // Objective Value

    Set_Info(&R, &I, &J, &K, "Data_cpp/info.bin");			// Initialize tensor size and rank from file

    cout << "R=" << R << ", I=" << I << ", J=" << J << ", K=" << K << endl;
	dims(0) = I, dims(1) = J, dims(2) = K;
	//------------------------------> Matrix Initializations <-------------------------------------

	MatrixXd A(I,R), B(J,R), C(K,R);						// Factors A,B,C
	MatrixXd A_T_A(R,R), B_T_B(R,R), C_T_C(R,R); 

	MatrixXd X_A(I, size_t(K * J));							// |
	MatrixXd X_B(J, size_t(K * I));							// | Tensor Matricization 
	MatrixXd X_C(K, size_t(I * J));							// |

	Read_Data(A, B, C, X_A, X_B, X_C, I, J, K, R);			// Read Factors and matricization from file

	//----------------------------> F_value Computation <------------------------------------------

	Solve_brasNN(A, B, C, X_A, X_B, X_C, I, J, K, R, block_size, AO_tol, max_iter, alpha_init, beta, f_value, AO_iter);
}