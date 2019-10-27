#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string>
#include "brascpd_functions.h"
#include <limits>


using namespace std;
using namespace Eigen;

int main(int argc, char **argv){

	VectorXi dims(3,1);
	int order = 3;											// Tensor Order
	int I, J, K;										    // Tensor Dimensions
	int R;												    // Rank of factorization
	

	const double AO_tol = 1e-3;							    // Tolerance for AO Algorithm
	int max_iter = 20000;									// Maximum Number of iterations
	int AO_iter  = 1;										// Iterations counter

	VectorXi block_size(3,1);								// |
	block_size.setConstant(400);							// | Parameters for stochastic
	const double alpha_init = 0.1;							// | gradient
	const double beta = 1e-4;								// | 
	VectorXi F_n(block_size(0),1);									// | fibers to be selected
	int factor;												// |
	VectorXi kr_idx(2,1);									// |

	double frob_X;				                		    // Frobenius norm of Tensor
	double f_value;										    // Objective Value

    Set_Info(&R, &I, &J, &K, "Data_cpp/info.bin");			// Initialize tensor size and rank from file

    cout << "R=" << R << ", I=" << I << ", J=" << J << ", K=" << K << endl;
	dims(0) = I, dims(1) = J, dims(2) = K;
	//------------------------------> Matrix Initializations <-------------------------------------

	MatrixXd A(I,R), B(J,R), C(K,R);						// Factors A,B,C
	MatrixXd A_T_A(R,R), B_T_B(R,R), C_T_C(R,R); 

	MatrixXd X_A_T(size_t(K * J), I);						// |
	MatrixXd X_B_T(size_t(K * I), J);						// | Tensor Matricization (transposed)
	MatrixXd X_C_T(size_t(I * J), K);						// |
	MatrixXd W_C(K, R);										// | MTTKRP X_C*kr(B,A) 

	Read_Data(A, B, C, X_A_T, X_B_T, X_C_T, I, J, K, R);	// Read Factors and matricization from file

	//----------------------------> F_value Computation <------------------------------------------

	frob_X = X_C_T.squaredNorm();							// Frob norm of tensor
	A_T_A.noalias() = A.transpose()*A;
	B_T_B.noalias() = B.transpose()*B;
	C_T_C.noalias() = C.transpose()*C;
	MatrixXd KhatriRao_BA(size_t(I * J),R);
	Khatri_Rao_Product(B, A, KhatriRao_BA);
	W_C = X_C_T.transpose()*KhatriRao_BA;

	f_value = Get_Objective_Value(C, X_C_T.transpose(), A_T_A, B_T_B, C_T_C, frob_X);

	cout << " BEGIN ALGORITHM " << endl;

	while (1)
	{
		cout << AO_iter << " -- " << f_value/sqrt(frob_X) << " -- " << f_value << " -- " << frob_X << " -- " <<  endl;
		Sampling_Operator(order, block_size, dims, F_n, kr_idx, factor);
		AO_iter++;
		return 0;
	}
	

}